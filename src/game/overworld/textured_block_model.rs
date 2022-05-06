use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model;
use crate::game::overworld::block_model::{BlockModel, ContentType, Vertex};
use crate::game::overworld::occluder::Occluder;
use crate::physics::aabb::AABB;
use bit_vec::BitVec;
use nalgebra_glm::{Vec2, Vec3};
use smallvec::{smallvec, SmallVec};
use std::ops::Range;

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

#[derive(Copy, Clone)]
pub struct QuadMaterial {
    material_id: u16,
    transparent: bool,
    texture_uv: QuadTexUV,
}

impl QuadMaterial {
    pub fn new(material_id: u16, transparent: bool, texture_uv: QuadTexUV) -> Self {
        Self {
            material_id,
            transparent,
            texture_uv,
        }
    }

    pub fn invisible() -> Self {
        Self {
            material_id: u16::MAX,
            transparent: true,
            texture_uv: QuadTexUV::new(Default::default()),
        }
    }

    pub fn transparent(&self) -> bool {
        self.transparent
    }
}

pub struct TexturedBlockModel {
    content_type: ContentType,
    vertices: Vec<Vertex>,
    side_quad_vertices: [Range<usize>; 6],
    inner_quad_vertices: Range<usize>,
    occluder: Occluder,
    quads_transparency: BitVec,
    aabbs: Vec<AABB>,
}

impl TexturedBlockModel {
    pub fn new(model: &BlockModel, quad_materials: &[QuadMaterial]) -> TexturedBlockModel {
        let mut tex_model = TexturedBlockModel {
            content_type: model.content_type(),
            vertices: vec![],
            side_quad_vertices: model.side_quads_range(),
            inner_quad_vertices: model.inner_quads_range(),
            occluder: model.occluder(),
            quads_transparency: quad_materials.iter().map(|m| m.transparent).collect(),
            aabbs: model.aabbs().to_vec(),
        };

        for (fi, range) in model.side_quads_range().iter().enumerate() {
            let facing = Facing::from_u8(fi as u8);

            for (i, q) in model.all_quads()[range.clone()].iter().enumerate() {
                if block_model::quad_occludes_side(q) && quad_materials[range.start + i].transparent() {
                    tex_model.occluder.clear_side(facing);
                    break;
                }
            }
        }

        tex_model.vertices = model
            .all_quads()
            .iter()
            .zip(quad_materials)
            .flat_map(|(quad, q_mat)| {
                let material_id = q_mat.material_id as u32;
                let vertices = quad.vertices();
                let normal = engine::utils::calc_triangle_normal(&vertices[0], &vertices[1], &vertices[2]);

                quad.vertices()
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
            })
            .collect();

        for s in &mut tex_model.side_quad_vertices {
            s.start *= 4;
            s.end *= 4;
        }
        tex_model.inner_quad_vertices.start *= 4;
        tex_model.inner_quad_vertices.end *= 4;

        tex_model
    }

    pub fn get_quads_by_facing(&self, facing: Facing) -> &[Vertex] {
        let range = self.side_quad_vertices[facing as usize].clone();
        &self.vertices[range]
    }

    pub fn get_inner_quads(&self) -> &[Vertex] {
        let range = self.inner_quad_vertices.clone();
        &self.vertices[range]
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
