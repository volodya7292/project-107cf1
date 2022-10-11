use std::ops::Range;

use approx::AbsDiffEq;
use glm::BVec3;
use nalgebra_glm as glm;
use nalgebra_glm::{U32Vec2, U32Vec3, UVec4, Vec2, Vec3};
use smallvec::{smallvec, SmallVec};

use engine::attributes_impl;
use engine::renderer::vertex_mesh::VertexPositionImpl;

use crate::overworld::facing::Facing;
use crate::overworld::occluder::Occluder;
use crate::overworld::raw_cluster;
use crate::overworld::raw_cluster::RawCluster;
use crate::physics::aabb::AABB;

const EQUITY_EPSILON: f32 = 1e-6;

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

#[derive(Debug, Default, Copy, Clone)]
pub struct Quad {
    vertices: [Vec3; 4],
}

impl Quad {
    /// Creates a new quad. Vertices must be ordered in triangle strip manner (Z).
    pub fn new(vertices: [Vec3; 4]) -> Quad {
        Quad { vertices }
    }

    pub fn vertices(&self) -> &[Vec3; 4] {
        &self.vertices
    }

    pub fn vertices_mut(&mut self) -> &mut [Vec3; 4] {
        &mut self.vertices
    }

    /// May return 0 if vertices are ordered incorrectly
    pub fn area(&self) -> f32 {
        let d0 = self.vertices[0] - self.vertices[1];
        let d1 = self.vertices[2] - self.vertices[1];
        let d2 = self.vertices[3] - self.vertices[1];

        (d0.cross(&d1) + d1.cross(&d2)).magnitude() / 2.0
    }

    /// Returns true if quads have the same shape. This is faster than `cmp_shape`, but quads vertices must be ordered.
    pub fn cmp_ordered(&self, other: &Self) -> bool {
        self.vertices
            .iter()
            .zip(&other.vertices)
            .all(|(v0, v1)| v0.abs_diff_eq(v1, EQUITY_EPSILON))
    }

    /// Returns true if quads have the same shape
    pub fn cmp_shape(&self, other: &Self) -> bool {
        self.vertices.iter().all(|v0| {
            other
                .vertices
                .iter()
                .any(|v1| v1.abs_diff_eq(&v0, EQUITY_EPSILON))
        })
    }
}

pub fn cube_quads(p0: Vec3, p1: Vec3) -> [Quad; 6] {
    let p000 = p0;
    let p001 = Vec3::new(p0.x, p0.y, p1.z);
    let p010 = Vec3::new(p0.x, p1.y, p0.z);
    let p011 = Vec3::new(p0.x, p1.y, p1.z);
    let p100 = Vec3::new(p1.x, p0.y, p0.z);
    let p101 = Vec3::new(p1.x, p0.y, p1.z);
    let p110 = Vec3::new(p1.x, p1.y, p0.z);
    let p111 = p1;

    [
        // Z
        Quad::new([p110, p100, p010, p000]),
        Quad::new([p011, p001, p111, p101]),
        // Y
        Quad::new([p000, p100, p001, p101]),
        Quad::new([p011, p111, p010, p110]),
        // X
        Quad::new([p010, p000, p011, p001]),
        Quad::new([p111, p101, p110, p100]),
    ]
}

#[derive(Default, Copy, Clone)]
pub struct ContentType(u8);

impl ContentType {
    const SOLID: Self = Self(0);
    const BINARY_TRANSPARENT: Self = Self(1);
    const TRANSLUCENT: Self = Self(2);
}

pub struct BlockModel {
    content_type: ContentType,
    quads: Vec<Quad>,
    side_quads_range: [Range<usize>; 6],
    inner_quads_range: Range<usize>,
    occluder: Occluder,
    aabbs: Vec<AABB>,
}

pub fn determine_quad_side(quad: &Quad) -> Option<Facing> {
    let v = quad.vertices()[0];

    // Find quad axis. If it's X axis, then v_eq = (1, 0, 0)
    let v_eq = quad.vertices().iter().fold(BVec3::from_element(true), |s, x| {
        glm::equal(&v, x).zip_map(&s, |a, b| a && b)
    });

    fn check(v: f32, f0: Facing, f1: Facing) -> Option<Facing> {
        if v.abs_diff_eq(&0.0, EQUITY_EPSILON) {
            Some(f0)
        } else if v.abs_diff_eq(&1.0, EQUITY_EPSILON) {
            Some(f1)
        } else {
            None
        }
    }

    // Determine side direction
    if v_eq.x {
        check(v.x, Facing::NegativeX, Facing::PositiveX)
    } else if v_eq.y {
        check(v.y, Facing::NegativeY, Facing::PositiveY)
    } else if v_eq.z {
        check(v.z, Facing::NegativeZ, Facing::PositiveZ)
    } else {
        None
    }
}

pub fn quad_occludes_side(quad: &Quad) -> bool {
    quad.area().abs_diff_eq(&1.0, EQUITY_EPSILON)
}

impl BlockModel {
    pub fn new(quads: &[Quad], aabbs: &[AABB]) -> BlockModel {
        let mut model = BlockModel {
            content_type: Default::default(),
            quads: vec![],
            side_quads_range: Default::default(),
            inner_quads_range: Default::default(),
            occluder: Default::default(),
            aabbs: aabbs.to_vec(),
        };

        let mut inner_quads = Vec::<Quad>::new();
        let mut side_quads = vec![Vec::<Quad>::new(); 6];

        for quad in quads {
            let facing = determine_quad_side(quad);

            if let Some(facing) = facing {
                side_quads[facing as usize].push(*quad);

                if quad_occludes_side(quad) {
                    model.occluder.occlude_side(facing);
                }
            } else {
                inner_quads.push(*quad);
            }
        }

        model.inner_quads_range = 0..inner_quads.len();
        model.quads.extend(inner_quads);

        let mut offset = model.inner_quads_range.end;

        for i in 0..6 {
            let end = offset + side_quads[i].len();
            model.side_quads_range[i] = offset..end;
            model.quads.extend(&side_quads[i]);
            offset = end;
        }

        model
    }

    pub fn content_type(&self) -> ContentType {
        self.content_type
    }

    pub fn occluder(&self) -> Occluder {
        self.occluder
    }

    pub fn get_quads_by_facing(&self, facing: Facing) -> &[Quad] {
        let range = self.side_quads_range[facing as usize].clone();
        &self.quads[range]
    }

    pub fn get_inner_quads(&self) -> &[Quad] {
        let range = self.inner_quads_range.clone();
        &self.quads[range]
    }

    pub fn aabbs(&self) -> &[AABB] {
        return &self.aabbs;
    }

    pub fn all_quads(&self) -> &[Quad] {
        &self.quads
    }

    pub fn inner_quads_range(&self) -> Range<usize> {
        self.inner_quads_range.clone()
    }

    pub fn side_quads_range(&self) -> [Range<usize>; 6] {
        self.side_quads_range.clone()
    }
    //
    // pub fn occludes_side_of_other(&self, other: &Self, facing: Facing) -> bool {
    //     let self_range = &self.side_quads_range[facing as usize];
    //     let other_range = &other.side_quads_range[facing.mirror() as usize];
    //     let mut occluded_other_quads: SmallVec<[bool; 32]> = smallvec![false; other_range.len()];
    //
    //     for (self_quad_i, self_quad) in self.quads[self_range.clone()].iter().enumerate() {
    //         let (mut self_min, mut self_max) = self_quad.vertices
    //             .iter()
    //             .fold((self_quad.vertices[0], self_quad.vertices[0]), |(min, max), v| {
    //                 (min.inf(&v), max.sup(&v))
    //             });
    //
    //         for (other_quad_i, other_quad) in other.quads[other_range.clone()].iter().enumerate()
    //         {
    //             if occluded_other_quads[other_quad_i] {
    //                 continue;
    //             }
    //
    //             let (other_min, other_max) = other_quad.vertices.iter().fold(
    //                 (other_quad.vertices[0], other_quad.vertices[0]),
    //                 |(min, max), v| (min.inf(&v), max.sup(&v)),
    //             );
    //
    //             // occluded_other_quads[other_quad_i] = (other_min - self_min).add_scalar(1e-6)
    //             //     >= Vec3::from_element(0.0)
    //             //     && (self_max - other_max).add_scalar(1e-6) >= Vec3::from_element(0.0);
    //         }
    //
    //        occluded_other_quads[] (self_min - self_max).abs() <= Vec3::from_element(1e-6)
    //     }
    //
    //     if all_transparent {
    //         // If all quads are transparent, they can't occlude anything
    //         return false;
    //     }
    //
    //     occluded_other_quads.iter().all(|v| *v)
    // }
}
