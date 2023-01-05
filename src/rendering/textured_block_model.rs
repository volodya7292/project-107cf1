use std::ops::Range;

use bit_vec::BitVec;
use nalgebra_glm as glm;
use nalgebra_glm::{U32Vec2, U32Vec3, UVec4, Vec2, Vec3};
use smallvec::{smallvec, SmallVec};

use core::overworld::block_model;
use core::overworld::block_model::{BlockModel, ContentType, Quad};
use core::overworld::facing::Facing;
use core::overworld::occluder::Occluder;
use core::overworld::raw_cluster::RawCluster;
use core::physics::aabb::AABB;
use core::registry::Registry;
use engine::attributes_impl;
use engine::renderer::vertex_mesh::VertexPositionImpl;

use crate::resource_mapping::ResourceMapping;

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct PackedVertex {
    pack1: UVec4,
    pack2: u32,
}
attributes_impl!(PackedVertex, pack1, pack2);

impl VertexPositionImpl for PackedVertex {
    fn position(&self) -> Vec3 {
        Vec3::new(
            ((self.pack1[0] >> 16) & 0xffff) as f32 / 65535.0 * (RawCluster::SIZE as f32),
            (self.pack1[0] & 0xffff) as f32 / 65535.0 * (RawCluster::SIZE as f32),
            ((self.pack1[1] >> 16) & 0xffff) as f32 / 65535.0 * (RawCluster::SIZE as f32),
        )
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tex_uv: Vec2,
    pub ao: u8,
    pub lighting: u16,
    pub material_id: u32,
}

impl Vertex {
    pub fn pack(&self) -> PackedVertex {
        let enc_pos: U32Vec3 = glm::convert_unchecked(self.position / (RawCluster::SIZE as f32) * 65535.0);
        let enc_normal: U32Vec3 =
            glm::convert_unchecked(glm::min(&(self.normal.add_scalar(1.0) * 0.5 * 255.0), 255.0));
        let enc_tex_uv: U32Vec2 = glm::convert_unchecked(self.tex_uv / 64.0 * 65535.0);

        let pack1_0 = ((enc_pos[0] & 0xffff) << 16) | (enc_pos[1] & 0xffff);
        let pack1_1 = ((enc_pos[2] & 0xffff) << 16) | ((enc_normal[0] & 0xff) << 8) | (enc_normal[1] & 0xff);
        let pack1_2 =
            ((enc_normal[2] & 0xff) << 24) | ((self.ao as u32 & 0xff) << 16) | (self.material_id & 0xffff);
        let pack1_3 = ((enc_tex_uv[0] & 0xffff) << 16) | (enc_tex_uv[1] & 0xffff);

        let pack2_4 = self.lighting as u32;

        PackedVertex {
            pack1: UVec4::new(pack1_0, pack1_1, pack1_2, pack1_3),
            pack2: pack2_4,
        }
    }
}

#[derive(Copy, Clone)]
pub enum QuadRotation {
    Rot0,
    Rot90,
    Rot180,
    Rot270,
}

#[derive(Copy, Clone)]
pub struct QuadTexUV([Vec2; 4]);

impl QuadTexUV {
    pub fn new(uv: [Vec2; 4]) -> QuadTexUV {
        QuadTexUV(uv)
    }

    pub fn rotate(&self, rotation: QuadRotation) -> QuadTexUV {
        let p = self.0;

        let v = match rotation {
            QuadRotation::Rot0 => [p[0], p[1], p[2], p[3]],
            QuadRotation::Rot90 => [p[2], p[0], p[3], p[1]],
            QuadRotation::Rot180 => [p[3], p[2], p[1], p[0]],
            QuadRotation::Rot270 => [p[1], p[3], p[0], p[2]],
        };

        QuadTexUV(v)
    }
}

impl Default for QuadTexUV {
    fn default() -> Self {
        QuadTexUV([
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
        ])
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct TexturedQuad {
    pub vertices: [Vertex; 4],
    pub transparent: bool,
}

#[derive(Copy, Clone)]
pub struct QuadMaterial {
    material_id: u16,
    texture_uv: QuadTexUV,
}

impl QuadMaterial {
    pub fn new(material_id: u16) -> Self {
        Self {
            material_id,
            texture_uv: Default::default(),
        }
    }

    pub fn with_tex_uv(mut self, texture_uv: QuadTexUV) -> Self {
        self.texture_uv = texture_uv;
        self
    }

    pub fn invisible() -> Self {
        Self {
            material_id: u16::MAX,
            texture_uv: QuadTexUV::new(Default::default()),
        }
    }
}

pub struct TexturedBlockModel {
    content_type: ContentType,
    quads: Vec<TexturedQuad>,
    side_quad_vertices: [Range<usize>; 6],
    first_side_quads_vsorted: [Quad; 6],
    inner_quad_vertices: Range<usize>,
    side_shapes_equality: [bool; 3],
    occluder: Occluder,
    quads_transparency: BitVec,
    merge_enabled: bool,
    aabbs: Vec<AABB>,
}

impl TexturedBlockModel {
    pub fn new(
        model: &BlockModel,
        quad_materials: &[QuadMaterial],
        res_map: &ResourceMapping,
    ) -> TexturedBlockModel {
        let mut tex_model = TexturedBlockModel {
            content_type: model.content_type(),
            quads: vec![],
            side_quad_vertices: model.side_quads_range(),
            first_side_quads_vsorted: Default::default(),
            inner_quad_vertices: model.inner_quads_range(),
            side_shapes_equality: [false; 3],
            occluder: model.occluder(),
            quads_transparency: quad_materials
                .iter()
                .map(|m| res_map.get_material(m.material_id).unwrap().translucent())
                .collect(),
            merge_enabled: true,
            aabbs: model.aabbs().to_vec(),
        };

        // Calculate general occluder
        for (f_i, range) in model.side_quads_range().iter().enumerate() {
            let facing = Facing::from_u8(f_i as u8);

            for (i, q) in model.all_quads()[range.clone()].iter().enumerate() {
                if i == 0 {
                    let centroid = q
                        .vertices()
                        .iter()
                        .fold(Vec3::from_element(0.0), |accum, v| accum + v)
                        / (q.vertices().len() as f32);

                    let mut quad_sorted = *q;
                    quad_sorted.vertices_mut().sort_by(|a, b| {
                        let (x_i, y_i) = match facing.axis_idx() {
                            0 => (1, 2),
                            1 => (0, 2),
                            2 => (0, 1),
                            _ => unreachable!(),
                        };
                        let a = Vec2::new(a[x_i], a[y_i]);
                        let b = Vec2::new(b[x_i], b[y_i]);

                        let a = (a.x - centroid.x).atan2(a.y - centroid.y);
                        let b = (b.x - centroid.x).atan2(b.y - centroid.y);
                        if facing.is_positive() {
                            a.total_cmp(&b)
                        } else {
                            b.total_cmp(&a)
                        }
                    });
                    tex_model.first_side_quads_vsorted[f_i] = quad_sorted;
                }

                let mat = res_map
                    .get_material(quad_materials[range.start + i].material_id)
                    .unwrap();

                if block_model::quad_occludes_side(q) && mat.translucent() {
                    tex_model.occluder.clear_side(facing);
                    break;
                }
            }
        }

        // Check if opposite side shapes are equal
        for (i, f) in [Facing::NegativeX, Facing::NegativeY, Facing::NegativeZ]
            .into_iter()
            .enumerate()
        {
            let inv_f = f.mirror();
            let neg_quads = model.get_quads_by_facing(f);
            let pos_quads = model.get_quads_by_facing(inv_f);

            let mut dir_mask: Vec3 = Vec3::from_element(1.0);
            dir_mask[f.axis_idx()] = 0.0;

            let equal = neg_quads.iter().all(|q0| {
                pos_quads.iter().any(|q1| {
                    let mut q0 = *q0;
                    let mut q1 = *q1;
                    let mut v0 = q0.vertices_mut();
                    let mut v1 = q1.vertices_mut();

                    for v in v0.iter_mut().chain(v1) {
                        *v = v.component_mul(&dir_mask);
                    }

                    q1.cmp_shape(&q0)
                })
            });

            tex_model.side_shapes_equality[i] = equal;
        }

        tex_model.quads = model
            .all_quads()
            .iter()
            .zip(quad_materials)
            .map(|(quad, q_mat)| {
                let material_id = q_mat.material_id as u32;
                let vertices = quad.vertices();
                let normal = core::utils::calc_triangle_normal(&vertices[0], &vertices[1], &vertices[2]);

                let tex_vertices: SmallVec<[_; 4]> = vertices
                    .iter()
                    .zip(&q_mat.texture_uv.0)
                    .map(move |(p, uv)| Vertex {
                        position: *p,
                        normal,
                        tex_uv: *uv,
                        ao: 0,
                        lighting: 0,
                        material_id,
                    })
                    .collect();

                let mat = res_map.get_material(q_mat.material_id).unwrap();

                TexturedQuad {
                    vertices: tex_vertices.into_inner().unwrap(),
                    transparent: mat.translucent(),
                }
            })
            .collect();

        tex_model
    }

    /// Specify whether to merge occluded blocks with the same model.
    pub fn with_merge(mut self, merge_enabled: bool) -> Self {
        self.merge_enabled = merge_enabled;
        self
    }

    pub fn merge_enabled(&self) -> bool {
        self.merge_enabled
    }

    /// Whether opposite sides on a given axis (X,Y,Z) have equal shapes.
    pub fn side_shapes_equality(&self) -> &[bool; 3] {
        &self.side_shapes_equality
    }

    pub fn first_side_quad_vsorted(&self, facing: Facing) -> &Quad {
        &self.first_side_quads_vsorted[facing as usize]
    }

    pub fn get_quads_by_facing(&self, facing: Facing) -> &[TexturedQuad] {
        let range = self.side_quad_vertices[facing as usize].clone();
        &self.quads[range]
    }

    pub fn get_inner_quads(&self) -> &[TexturedQuad] {
        let range = self.inner_quad_vertices.clone();
        &self.quads[range]
    }

    pub fn occluder(&self) -> Occluder {
        // TODO: account for texture transparency
        self.occluder
    }

    pub fn aabbs(&self) -> &[AABB] {
        &self.aabbs
    }

    pub fn is_opaque(&self) -> bool {
        self.occluder.is_full()
    }
}
