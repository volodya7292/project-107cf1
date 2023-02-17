use crate::client::overworld::raw_cluster_ext::{ClientRawCluster, ClusterMeshes};
use crate::default_resources::DefaultResourceMapping;
use crate::game::Game;
use crate::resource_mapping::ResourceMapping;
use base::overworld::orchestrator::get_side_clusters;
use base::overworld::orchestrator::OverworldUpdateResult;
use base::overworld::position::ClusterPos;
use base::overworld::LoadedClusters;
use common::glm::{DVec3, I64Vec3, Vec3};
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::types::{HashMap, HashSet};
use common::{glm, rayon};
use engine::ecs::component;
use engine::ecs::component::{MeshRenderConfigC, TransformC, VertexMeshC};
use engine::module::main_renderer::{MainRenderer, VertexMeshObject};
use engine::module::scene::Scene;
use engine::vkw;
use entity_data::EntityId;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct OverworldRenderer {
    device: Arc<vkw::Device>,
    cluster_mat_pipeline: u32,
    resource_mapping: Arc<ResourceMapping>,
    loaded_clusters: LoadedClusters,
    root_entity: EntityId,

    to_add: HashSet<ClusterPos>,
    to_remove: HashSet<ClusterPos>,
    to_build_mesh: Mutex<HashSet<ClusterPos>>,
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
        root_entity: EntityId,
    ) -> Self {
        Self {
            device,
            cluster_mat_pipeline,
            resource_mapping,
            loaded_clusters,
            root_entity,
            to_add: HashSet::with_capacity(8192),
            to_remove: HashSet::with_capacity(8192),
            to_build_mesh: Mutex::new(HashSet::with_capacity(8192)),
            to_update_mesh: Mutex::new(HashMap::with_capacity(1024)),
            entities: HashMap::with_capacity(8192),
        }
    }

    pub fn manage_changes(&mut self, overworld_update: &OverworldUpdateResult) {
        let mut to_build_mesh = self.to_build_mesh.lock();

        to_build_mesh.extend(overworld_update.processed_dirty_clusters.iter().map(|v| v.0));
        to_build_mesh.extend(overworld_update.updated_auxiliary_parts.iter().map(|v| v.0));

        // Handle new clusters
        for pos in &overworld_update.new_clusters {
            self.to_add.insert(*pos);
            self.to_remove.remove(pos);
            to_build_mesh.insert(*pos);
        }

        // Handle removed clusters
        for pos in overworld_update
            .removed_clusters
            .iter()
            .chain(&overworld_update.offloaded_clusters)
        {
            self.to_remove.insert(*pos);
            self.to_add.remove(pos);
            to_build_mesh.remove(pos);
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
        let mut to_build_sorted: Vec<_> = self.to_build_mesh.lock().iter().cloned().collect();
        to_build_sorted.par_sort_by_cached_key(|pos| {
            let diff = pos.get() - stream_pos_i;
            diff.dot(&diff)
        });

        // Build new meshes for dirty clusters
        for chunk in to_build_sorted.chunks(rayon::current_num_threads()) {
            chunk.par_iter().for_each(|pos| {
                let o_cluster = o_clusters.get(pos).unwrap();
                if !o_cluster.state().is_loaded() {
                    return;
                }
                let t_cluster_guard = o_cluster.cluster.read();
                let Some(t_cluster) = t_cluster_guard.as_ref() else {
                    // Cluster is not accessible because it must be offloaded
                    assert!(o_cluster.state().is_offloaded());
                    return;
                };

                let meshes = t_cluster.raw().build_mesh(&self.device, &self.resource_mapping);
                self.to_update_mesh.lock().insert(*pos, meshes);
                self.to_build_mesh.lock().remove(pos);
            });

            let t1 = Instant::now();
            if t1.duration_since(t_start) >= max_execution_time {
                break;
            }
        }
    }

    pub fn update_scene(&mut self, scene: &mut Scene) {
        let mut to_update_mesh = self.to_update_mesh.lock();

        // Remove objects
        for pos in self.to_remove.drain() {
            if let Some(entities) = self.entities.remove(&pos) {
                scene.remove_object(&entities.solid);
                scene.remove_object(&entities.translucent);
            }
            to_update_mesh.remove(&pos);
        }

        let to_build = self.to_build_mesh.lock();

        // Update meshes for scene objects
        to_update_mesh.retain(|pos, meshes| {
            for neighbour in get_side_clusters(&pos) {
                if to_build.contains(&neighbour) {
                    // Retain, do mesh update later when neighbours are ready
                    return true;
                }
            }

            let entities = self.entities.entry(*pos).or_insert_with(|| {
                assert_eq!(self.to_add.remove(pos), true);

                let transform_comp = TransformC::new().with_position(glm::convert(*pos.get()));
                let render_config_solid = MeshRenderConfigC::new(self.cluster_mat_pipeline, false);
                let render_config_translucent = MeshRenderConfigC::new(self.cluster_mat_pipeline, true);

                let entity_solid = scene.add_object(
                    Some(self.root_entity),
                    VertexMeshObject::new(transform_comp, render_config_solid, Default::default()),
                );
                let entity_translucent = scene.add_object(
                    Some(self.root_entity),
                    VertexMeshObject::new(transform_comp, render_config_translucent, Default::default()),
                );

                ClusterEntities {
                    solid: entity_solid.unwrap(),
                    translucent: entity_translucent.unwrap(),
                }
            });

            if let Some(mut entry) = scene.entry(&entities.solid) {
                *entry.get_mut::<VertexMeshC>().unwrap() = VertexMeshC::new(&meshes.solid.raw());
            }
            if let Some(mut entry) = scene.entry(&entities.translucent) {
                *entry.get_mut::<VertexMeshC>().unwrap() = VertexMeshC::new(&meshes.transparent.raw());
            }

            false
        });
    }
}
