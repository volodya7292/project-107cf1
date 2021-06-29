use crate::game::overworld::block_component::Facing;
use approx::AbsDiffEq;
use glm::BVec3;
use nalgebra_glm as glm;
use nalgebra_glm::{Vec2, Vec3};
use std::ops::Range;

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tex_uv: Vec2,
    pub ao: f32,
    pub material_id: u32,
}
vertex_impl!(Vertex, position, normal, tex_uv, ao, material_id);

#[derive(Debug, Copy, Clone)]
pub struct Quad {
    vertices: [Vec3; 4],
}

impl Quad {
    pub fn new(vertices: [Vec3; 4]) -> Quad {
        Quad { vertices }
    }

    pub fn vertices(&self) -> &[Vec3; 4] {
        &self.vertices
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
    pub(crate) quads: Vec<Quad>,
    pub(crate) side_quads: [Range<usize>; 6],
    pub(crate) inner_quads: Range<usize>,
    pub(crate) occluded_sides: [bool; 6],
}

fn determine_quad_side(quad: &Quad) -> Option<Facing> {
    let v = quad.vertices()[0];
    let v_eq = quad.vertices().iter().fold(BVec3::from_element(true), |s, x| {
        glm::equal(&v, x).zip_map(&s, |a, b| a && b)
    });

    fn check(v: f32, f0: Facing, f1: Facing) -> Option<Facing> {
        if v.abs_diff_eq(&0.0, 1e-7) {
            Some(f0)
        } else if v.abs_diff_eq(&1.0, 1e-7) {
            Some(f1)
        } else {
            None
        }
    }

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

impl BlockModel {
    pub fn new(quads: &[Quad]) -> BlockModel {
        let mut model = BlockModel {
            content_type: Default::default(),
            quads: vec![],
            side_quads: Default::default(),
            inner_quads: Default::default(),
            occluded_sides: Default::default(),
        };

        let mut inner_quads = Vec::<Quad>::new();
        let mut side_quads = vec![Vec::<Quad>::new(); 6];

        for quad in quads {
            let facing = determine_quad_side(quad);

            if let Some(facing) = facing {
                side_quads[facing as usize].push(*quad);
            } else {
                inner_quads.push(*quad);
            }
        }

        model.inner_quads = 0..inner_quads.len();
        model.quads.extend(inner_quads);

        let mut offset = model.inner_quads.end;

        for i in 0..6 {
            let end = offset + side_quads[i].len();
            model.side_quads[i] = offset..end;
            model.quads.extend(&side_quads[i]);
            offset = end;
        }

        model
    }

    pub fn content_type(&self) -> ContentType {
        self.content_type
    }

    pub fn occludes_side(&self, facing: Facing) -> bool {
        self.occluded_sides[facing as usize]
    }

    pub fn get_quads_by_facing(&self, facing: Facing) -> &[Quad] {
        let range = self.side_quads[facing as usize].clone();
        &self.quads[range]
    }

    pub fn get_inner_quads(&self) -> &[Quad] {
        let range = self.inner_quads.clone();
        &self.quads[range]
    }
}