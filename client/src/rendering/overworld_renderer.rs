use crate::client::overworld::raw_cluster_ext::{ClusterMeshBuilder, ClusterMeshes};
use crate::resource_mapping::ResourceMapping;
use base::execution;
use base::execution::virtual_processor::{VirtualProcessor, VirtualTask};
use base::execution::{default_queue, Task};
use base::overworld::accessor::ClusterNeighbourhoodAccessor;
use base::overworld::cluster_part_set::{part_idx_to_dir, ClusterPartSet};
use base::overworld::orchestrator::get_side_clusters;
use base::overworld::orchestrator::OverworldUpdateResult;
use base::overworld::position::ClusterPos;
use base::overworld::{ClusterStateEnum, LoadedClusters};
use base::registry::Registry;
use common::glm::{DVec3, I64Vec3};
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::threading::SafeThreadPool;
use common::tokio::sync::Notify;
use common::types::{HashMap, HashSet};
use common::{glm, MO_RELAXED};
use engine::ecs::component::{MeshRenderConfigC, TransformC, VertexMeshC};
use engine::module::main_renderer::VertexMeshObject;
use engine::module::scene::Scene;
use engine::vkw;
use entity_data::EntityId;
use smallvec::SmallVec;
use std::collections::hash_map;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct OverworldRenderer {
    default_queue: Arc<SafeThreadPool>,
    device: Arc<vkw::Device>,
    cluster_mat_pipeline: u32,
    registry: Arc<Registry>,
    resource_mapping: Arc<ResourceMapping>,
    loaded_clusters: LoadedClusters,
    root_entity: EntityId,

    to_remove: HashSet<ClusterPos>,
    to_build_meshes: HashMap<ClusterPos, ClusterPartSet>,
    to_update_meshes: Arc<Mutex<HashMap<ClusterPos, ClusterMeshes>>>,

    entities: HashMap<ClusterPos, ClusterEntities>,
    r_clusters: HashMap<ClusterPos, RCluster>,
}

#[derive(Default)]
struct BooleanFlag {
    event: Notify,
    value: AtomicBool,
}

impl BooleanFlag {
    fn signal(&self) {
        self.value.store(true, MO_RELAXED);
        self.event.notify_waiters();
    }

    async fn wait(&self) {
        while !self.value.load(MO_RELAXED) {
            self.event.notified().await;
        }
    }
}

struct PendingUpdate {
    pos: ClusterPos,
    dependencies: SmallVec<[Arc<BooleanFlag>; ClusterPartSet::NUM]>,
    finish_event: Arc<BooleanFlag>,

    to_update_meshes: Arc<Mutex<HashMap<ClusterPos, ClusterMeshes>>>,
    registry: Arc<Registry>,
    res_map: Arc<ResourceMapping>,
    loaded_clusters: LoadedClusters,
    device: Arc<vkw::Device>,

    build_task: VirtualTask<Option<ClusterMeshes>>,
}

struct RCluster {
    build_mesh_task: Option<Task<()>>,
    finish_event: Arc<BooleanFlag>,
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

async fn build_cluster_mesh(update: PendingUpdate) {
    // 1. Perform the update
    let meshes = update.build_task.future().await;

    // 2. Notify others
    update.finish_event.signal();

    // 3. Wait for dependencies clusters to update
    for dep in &update.dependencies {
        dep.wait().await;
    }

    // 4. Commit the mesh
    if let Some(meshes) = meshes {
        update.to_update_meshes.lock().insert(update.pos, meshes);
    };
}

impl OverworldRenderer {
    pub fn new(
        device: Arc<vkw::Device>,
        cluster_mat_pipeline: u32,
        registry: Arc<Registry>,
        resource_mapping: Arc<ResourceMapping>,
        loaded_clusters: LoadedClusters,
        root_entity: EntityId,
    ) -> Self {
        Self {
            default_queue: default_queue().unwrap(),
            device,
            cluster_mat_pipeline,
            registry,
            resource_mapping,
            loaded_clusters,
            root_entity,
            to_remove: HashSet::with_capacity(8192),
            to_build_meshes: HashMap::with_capacity(8192),
            to_update_meshes: Arc::new(Mutex::new(HashMap::with_capacity(1024))),
            entities: HashMap::with_capacity(8192),
            r_clusters: HashMap::with_capacity(8192),
        }
    }

    pub fn manage_changes(&mut self, overworld_update: &OverworldUpdateResult) {
        // Handle new clusters
        for pos in &overworld_update.new_clusters {
            self.r_clusters.insert(
                *pos,
                RCluster {
                    build_mesh_task: None,
                    finish_event: Default::default(),
                },
            );
            self.to_remove.remove(pos);
        }

        let to_remove_iter = overworld_update.removed_clusters.iter();

        // Handle removed clusters
        for pos in to_remove_iter.clone() {
            let mut r_cl = self.r_clusters.remove(pos).unwrap();
            if let Some(task) = r_cl.build_mesh_task.take() {
                task.cancel();
            }
            r_cl.finish_event.signal();

            self.to_remove.insert(*pos);
            self.to_build_meshes.remove(pos);
        }

        // Nearby clusters must be updated if the central cluster is removed
        for pos in to_remove_iter {
            for rel_pos in get_side_clusters(pos) {
                if !self.r_clusters.contains_key(&rel_pos) {
                    continue;
                }
                if let hash_map::Entry::Vacant(e) = self.to_build_meshes.entry(rel_pos) {
                    e.insert(ClusterPartSet::NONE);
                }
            }
        }
    }

    pub fn update(&mut self, stream_pos: DVec3, overworld_update: &OverworldUpdateResult) {
        self.manage_changes(overworld_update);

        let stream_pos_i: I64Vec3 = glm::convert_unchecked(stream_pos);
        let o_clusters = self.loaded_clusters.read();

        // 1. Accumulate updates
        for (pos, parts) in &overworld_update.dirty_clusters_parts {
            let curr_parts = self.to_build_meshes.entry(*pos).or_default();
            *curr_parts |= *parts;
        }

        // 2. Collect dependent clusters, meshes of which may have been affected by changes of this cluster
        for (pos, parts) in self.to_build_meshes.clone() {
            for part_idx in parts.iter_ones() {
                let dir = part_idx_to_dir(part_idx);
                let rel_pos = pos.offset_i32(&dir);

                if !o_clusters
                    .get(&rel_pos)
                    .is_some_and(|o_cluster| o_cluster.state().is_loaded())
                {
                    continue;
                }

                if let hash_map::Entry::Vacant(e) = self.to_build_meshes.entry(rel_pos) {
                    e.insert(ClusterPartSet::NONE);
                }
            }
        }

        // 3. Collect cluster meshes to build
        let to_build_meshes: HashSet<_> = self
            .to_build_meshes
            .keys()
            .cloned()
            .filter(|pos| {
                let r_cl = self.r_clusters.get_mut(pos).unwrap();
                let prev_task_finished = r_cl.build_mesh_task.as_ref().map_or(true, |v| v.is_finished());

                if prev_task_finished {
                    r_cl.build_mesh_task = None;
                    r_cl.finish_event = Default::default();
                }

                // If any neighbour is not loaded yet, do not build the mesh of this cluster
                if get_side_clusters(pos)
                    .iter()
                    .any(|rel_pos| o_clusters.get(rel_pos).map_or(false, |v| !v.state().is_loaded()))
                {
                    return false;
                }

                prev_task_finished
            })
            .collect();

        // Sort dirty clusters by distance from observer
        let mut sorted_build_positions: Vec<_> = to_build_meshes.iter().cloned().collect();
        sorted_build_positions.par_sort_by_cached_key(|pos| {
            let diff = pos.get() - stream_pos_i;
            diff.dot(&diff)
        });

        // 4. Schedule new updates
        let build_processor = VirtualProcessor::new(&self.default_queue);

        for pos in &sorted_build_positions {
            let r_cl = self.r_clusters.get(pos).unwrap();

            let deps: SmallVec<[ClusterPos; ClusterPartSet::NUM]> = get_side_clusters(pos)
                .into_iter()
                .filter(|v| to_build_meshes.contains(v))
                .collect();

            let build_task = {
                let registry = self.registry.clone();
                let loaded_clusters = self.loaded_clusters.clone();
                let pos = *pos;
                let res_map = self.resource_mapping.clone();
                let device = self.device.clone();

                build_processor.spawn(move || {
                    let accessor = ClusterNeighbourhoodAccessor::new(registry, &loaded_clusters, pos);
                    accessor
                        .is_center_available()
                        .then(|| accessor.build_mesh(&device, &res_map))
                })
            };

            let update_info = PendingUpdate {
                pos: *pos,
                dependencies: deps
                    .iter()
                    .map(|v| Arc::clone(&self.r_clusters[v].finish_event))
                    .collect(),
                finish_event: Arc::clone(&r_cl.finish_event),
                to_update_meshes: Arc::clone(&self.to_update_meshes),
                registry: Arc::clone(&self.registry),
                res_map: Arc::clone(&self.resource_mapping),
                loaded_clusters: Arc::clone(&self.loaded_clusters),
                device: Arc::clone(&self.device),
                build_task,
            };

            let build_mesh_task = execution::spawn_coroutine(build_cluster_mesh(update_info));

            let r_cl = self.r_clusters.get_mut(pos).unwrap();
            r_cl.build_mesh_task = Some(build_mesh_task);
        }
        build_processor.detach();

        // Remove schedules clusters
        self.to_build_meshes
            .retain(|pos, _| !to_build_meshes.contains(pos));
    }

    pub fn update_scene(&mut self, scene: &mut Scene) {
        let mut to_update_mesh = self.to_update_meshes.lock();

        // Remove objects
        for pos in self.to_remove.drain() {
            if let Some(entities) = self.entities.remove(&pos) {
                scene.remove_object(&entities.solid);
                scene.remove_object(&entities.translucent);
            }
            to_update_mesh.remove(&pos);
        }

        // Update meshes for scene objects
        for (pos, meshes) in to_update_mesh.drain() {
            if meshes.solid.vertex_count() == 0 && meshes.transparent.vertex_count() == 0 {
                if let Some(entities) = self.entities.remove(&pos) {
                    scene.remove_object(&entities.solid);
                    scene.remove_object(&entities.translucent);
                }
                continue;
            }

            let entities = self.entities.entry(pos).or_insert_with(|| {
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

            {
                let mut entry = scene.entry(&entities.solid);
                *entry.get_mut_checked::<VertexMeshC>().unwrap() = VertexMeshC::new(&meshes.solid.raw());
            }
            {
                let mut entry = scene.entry(&entities.translucent);
                *entry.get_mut_checked::<VertexMeshC>().unwrap() =
                    VertexMeshC::new(&meshes.transparent.raw());
            }
        }
    }
}
