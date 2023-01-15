use std::sync::Arc;
use std::time::{Duration, Instant};

use entity_data::EntityId;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, Vec3};
use parking_lot::Mutex;
use rayon::prelude::*;

use core::overworld::orchestrator::get_side_clusters;
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

    pub fn manage_changes(&mut self, overworld_update: &OverworldUpdateResult) {
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
    }

    pub fn update(
        &mut self,
        stream_pos: DVec3,
        overworld_update: &OverworldUpdateResult,
        max_execution_time: Duration,
    ) {
        let t_start = Instant::now();

        self.manage_changes(overworld_update);

        let stream_pos_i: I64Vec3 = glm::convert_unchecked(stream_pos);
        let o_clusters = self.loaded_clusters.read();

        // Sort dirty clusters by distance from observer
        let mut dirty_clusters_sorted: Vec<_> = self.dirty_clusters.iter().cloned().collect();
        dirty_clusters_sorted.par_sort_by_cached_key(|pos| {
            let diff = pos.get() - stream_pos_i;
            diff.dot(&diff)
        });

        // Build new meshes for dirty clusters
        for chunk in dirty_clusters_sorted.chunks(rayon::current_num_threads()) {
            chunk.par_iter().for_each(|pos| {
                let o_cluster = o_clusters.get(pos).unwrap();
                let t_cluster_guard = o_cluster.cluster.read();
                let t_cluster = t_cluster_guard.as_ref().unwrap();

                let meshes = t_cluster.raw().build_mesh(&self.device, &self.resource_mapping);
                self.to_update_mesh.lock().insert(*pos, meshes);
            });

            for pos in chunk {
                self.dirty_clusters.remove(pos);
            }

            let t1 = Instant::now();
            if t1.duration_since(t_start) >= max_execution_time {
                break;
            }
        }
    }

    pub fn update_scene(&mut self, renderer: &mut Renderer) {
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

        // Update meshes for scene objects
        to_update_mesh.retain(|pos, meshes| {
            let entities = self.entities.get(&pos).unwrap();

            for neighbour in get_side_clusters(&pos) {
                if self.dirty_clusters.contains(&neighbour) {
                    // Retain, do mesh update later when neighbours are ready
                    return true;
                }
            }

            *renderer
                .access_object(entities.solid)
                .get_mut::<component::VertexMesh>()
                .unwrap() = component::VertexMesh::new(&meshes.solid.raw());

            *renderer
                .access_object(entities.translucent)
                .get_mut::<component::VertexMesh>()
                .unwrap() = component::VertexMesh::new(&meshes.transparent.raw());

            false
        });
    }
}
