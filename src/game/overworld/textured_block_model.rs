use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{BlockModel, ContentType, Vertex};
use nalgebra_glm::Vec2;
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
    texture_uv: QuadTexUV,
}

impl QuadMaterial {
    pub fn new(material_id: u16, texture_uv: QuadTexUV) -> QuadMaterial {
        QuadMaterial {
            material_id,
            texture_uv,
        }
    }
}

pub struct TexturedBlockModel {
    content_type: ContentType,
    vertices: Vec<Vertex>,
    side_quad_vertices: [Range<usize>; 6],
    inner_quad_vertices: Range<usize>,
    occluded_sides: [bool; 6],
}

impl TexturedBlockModel {
    pub fn new(block_model: &BlockModel, quad_materials: &[QuadMaterial]) -> TexturedBlockModel {
        let mut model = TexturedBlockModel {
            content_type: block_model.content_type(),
            vertices: vec![],
            side_quad_vertices: block_model.side_quads.clone(),
            inner_quad_vertices: block_model.inner_quads.clone(),
            occluded_sides: block_model.occluded_sides.clone(),
        };

        model.vertices = block_model
            .quads
            .iter()
            .zip(quad_materials)
            .flat_map(|(quad, q_mat)| {
                let material_id = q_mat.material_id as u32;

                quad.vertices()
                    .iter()
                    .zip(&q_mat.texture_uv.0)
                    .map(move |(p, uv)| Vertex {
                        position: *p,
                        normal: Default::default(), // TODO
                        tex_uv: *uv,
                        ao: 0.0, // TODO
                        material_id,
                    })
            })
            .collect();

        for s in &mut model.side_quad_vertices {
            s.start *= 4;
            s.end *= 4;
        }
        model.inner_quad_vertices.start *= 4;
        model.inner_quad_vertices.end *= 4;

        model
    }

    pub fn get_quads_by_facing(&self, facing: Facing) -> &[Vertex] {
        let range = self.side_quad_vertices[facing as usize].clone();
        &self.vertices[range]
    }

    pub fn get_inner_quads(&self) -> &[Vertex] {
        let range = self.inner_quad_vertices.clone();
        &self.vertices[range]
    }
}
