use std::sync::Arc;
use std::time::{Duration, Instant};

use entity_data::EntityId;
use nalgebra_glm as glm;
use nalgebra_glm::Vec3;
use parking_lot::Mutex;
use rayon::prelude::*;

use core::overworld::orchestrator::OverworldUpdateResult;
use core::overworld::position::ClusterPos;
use core::overworld::LoadedClusters;
use core::utils::{HashMap, HashSet};
use engine::ecs::component;
use engine::renderer::{Renderer, VertexMeshObject};
use vk_wrapper as vkw;

use crate::client::overworld::raw_cluster_ext::{ClientRawCluster, ClusterMeshes};
use crate::default_resources::DefaultResourceMapping;
use crate::resource_mapping::ResourceMapping;

pub struct OverworldRenderer {
    device: Arc<vkw::Device>,
    cluster_mat_pipeline: u32,
    resource_mapping: Arc<ResourceMapping>,
    loaded_clusters: LoadedClusters,

    dirty_clusters: HashSet<ClusterPos>,
    to_add: HashSet<ClusterPos>,
    to_remove: HashSet<ClusterPos>,
    to_update_mesh: Mutex<HashMap<ClusterPos, ClusterMeshes>>,

    entities: HashMap<ClusterPos, ClusterEntities>,
    // /// Whether `edge_intrinsics` has been determined, only then `occluded` can be set
    // edge_intrinsics_determined: Arc<AtomicBool>,
    // /// All conditions are met for the vertex mesh to be updated
    // mesh_can_be_updated: Arc<AtomicBool>,
    // /// Mesh has been updated and it needs to be updated inside the VertexMesh scene component
    // mesh_changed: Arc<AtomicBool>,
}

#[derive(Default)]
struct ClusterEntities {
    solid: EntityId,
    translucent: EntityId,
}

impl ClusterEntities {
    fn is_null(&self) -> bool {
        self.solid == EntityId::NULL && self.translucent == EntityId::NULL
    }
}

impl OverworldRenderer {
    pub fn new(
        device: Arc<vkw::Device>,
        cluster_mat_pipeline: u32,
        resource_mapping: Arc<ResourceMapping>,
        loaded_clusters: LoadedClusters,
    ) -> Self {
        Self {
            device,
            cluster_mat_pipeline,
            resource_mapping,
            loaded_clusters,
            dirty_clusters: HashSet::with_capacity(8192),
            to_add: HashSet::with_capacity(8192),
            to_remove: HashSet::with_capacity(8192),
            to_update_mesh: Mutex::new(HashMap::with_capacity(1024)),
            entities: HashMap::with_capacity(8192),
        }
    }

    pub fn update(&mut self, overworld_update: &OverworldUpdateResult, max_execution_time: Duration) {
        self.dirty_clusters
            .extend(overworld_update.processed_dirty_clusters.iter().map(|v| v.0));

        self.dirty_clusters
            .extend(overworld_update.updated_auxiliary_parts.iter().map(|v| v.0));

        for pos in &overworld_update.new_clusters {
            self.to_add.insert(*pos);
            self.to_remove.remove(pos);
        }

        for gov in &overworld_update.processed_dirty_clusters {
            assert!(!self.to_remove.contains(&gov.0));
        }
        for gov in &overworld_update.updated_auxiliary_parts {
            assert!(!self.to_remove.contains(&gov.0));
        }

        for pos in overworld_update
            .removed_clusters
            .iter()
            .chain(&overworld_update.offloaded_clusters)
        {
            self.to_remove.insert(*pos);
            self.to_add.remove(pos);
            self.dirty_clusters.remove(pos);
        }

        // -----------------------------------------------------------------------------------------------
        let o_clusters = self.loaded_clusters.read();

        let to_build_meshes: Vec<_> = self
            .dirty_clusters
            .iter()
            .cloned()
            // TODO
            // .take(max_mesh_updates)
            .take(10)
            .collect();

        for pos in &to_build_meshes {
            self.dirty_clusters.remove(pos);
        }

        to_build_meshes.par_iter().for_each(|pos| {
            let o_cluster = o_clusters.get(pos).unwrap();
            if !o_cluster.state().is_loaded() {
                println!("{:?} {}", o_cluster.state(), o_cluster.state().is_offloaded());
            }
            let t_cluster_guard = o_cluster.cluster.read();
            let t_cluster = t_cluster_guard.as_ref().unwrap();

            let meshes = t_cluster.raw.build_mesh(&self.device, &self.resource_mapping);

            self.to_update_mesh.lock().insert(*pos, meshes);
        });
    }

    pub fn update_scene(&mut self, renderer: &mut Renderer) {
        // TODO: remove
        for gov in &self.to_remove {
            assert!(!self.dirty_clusters.contains(gov));
        }

        let mut to_update_mesh = self.to_update_mesh.lock();

        // Remove objects
        for pos in self.to_remove.drain() {
            if let Some(entities) = self.entities.remove(&pos) {
                renderer.remove_object(&entities.solid);
                renderer.remove_object(&entities.translucent);
            }
            to_update_mesh.remove(&pos);
        }

        // Add new objects
        for pos in self.to_add.drain() {
            let transform_comp = component::Transform::new(
                glm::convert(*pos.get()),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let render_config_solid = component::MeshRenderConfig::new(self.cluster_mat_pipeline, false);
            let render_config_translucent = component::MeshRenderConfig::new(self.cluster_mat_pipeline, true);

            let entity_solid = renderer.add_object(VertexMeshObject::new(
                transform_comp,
                render_config_solid,
                Default::default(),
            ));
            let entity_translucent = renderer.add_object(VertexMeshObject::new(
                transform_comp,
                render_config_translucent,
                Default::default(),
            ));

            self.entities.insert(
                pos,
                ClusterEntities {
                    solid: entity_solid,
                    translucent: entity_translucent,
                },
            );
        }

        // Update meshes
        for (pos, meshes) in to_update_mesh.drain() {
            let entities = self.entities.get(&pos).unwrap();

            *renderer
                .access_object(entities.solid)
                .get_mut::<component::VertexMesh>()
                .unwrap() = component::VertexMesh::new(&meshes.solid.raw());

            *renderer
                .access_object(entities.translucent)
                .get_mut::<component::VertexMesh>()
                .unwrap() = component::VertexMesh::new(&meshes.transparent.raw());
        }
    }
}
