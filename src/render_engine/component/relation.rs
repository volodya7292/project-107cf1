use crate::render_engine::scene::Entity;
use crate::utils::{HashSet, IndexSet};

pub struct Parent(pub(in crate::render_engine) Entity);

#[derive(Default)]
pub struct Children {
    children: IndexSet<Entity>,
    preserve_order: bool,
}

impl Children {
    pub fn new(children: Vec<Entity>, preserve_order: bool) -> Self {
        Children {
            children: IndexSet::from_iter(children),
            preserve_order,
        }
    }

    pub fn get(&self) -> &IndexSet<Entity> {
        &self.children
    }
}

// ----------------------------------------------------------------------------------------------------
// Note: children usage example
// ----------------------------------------------------------------------------------------------------
//
// let transform_comps = &mut data.transform;
// let renderer_comps = &mut data.renderer;
// let vertex_mesh_comps = &mut data.vertex_mesh;
// let parent_comps = &mut data.parent;
// let children_comps = &mut data.children;
// let entities = &mut data.entities;
//
// let is_children_empty = if let Some(children) = children_comps.get(entity) {
//     children.get().is_empty()
// } else {
//     children_comps.set(entity, component::Children::default());
//     true
// };
//
// if is_children_empty {
//     let sector_count = SIZE_IN_SECTORS * SIZE_IN_SECTORS * SIZE_IN_SECTORS;
//     let children: Vec<u32> = (0..sector_count).into_iter().map(|_| entities.create()).collect();
//
//     component::set_children(entity, &children, parent_comps, children_comps);
//
//     for (i, &ent) in children.iter().enumerate() {
//         let p = index_1d_to_3d(i, SIZE_IN_SECTORS);
//         let node_size = self.entry_size as usize;
//
//         transform_comps.set(
//             ent,
//             component::Transform::new(
//                 glm::convert(TVec3::new(p[0], p[1], p[2]) * SECTOR_SIZE * node_size),
//                 Vec3::default(),
//                 Vec3::from_element(1.0),
//             ),
//         );
//
//         let renderer = component::Renderer::new(&renderer, data.mat_pipeline, false);
//         renderer_comps.set(ent, renderer);
//     }
// }
//
// let children = children_comps.get(entity).unwrap();
//
// for (i, &ent) in children.get().iter().enumerate() {
//     let sector = &self.sectors[i];
//
//     vertex_mesh_comps.set(ent, component::VertexMesh::new(&sector.vertex_mesh.raw()));
// }
