use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::{component, Renderer};
use crate::utils::{HashMap, HashSet};
use crate::world::block_component::Facing;
use crate::world::block_registry::BlockRegistry;
use crate::world::cluster;
use crate::world::cluster::Cluster;
use crate::world::generator;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, Vec3};
use rayon::prelude::*;
use simdnoise::NoiseBuilder;
use smallvec::SmallVec;
use std::collections::hash_map;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const MAX_LOD: usize = 4;
pub const LOD0_RANGE: usize = 128;

struct WorldCluster {
    cluster: Arc<Mutex<Cluster>>,
    entity: u32,
    creation_time_secs: u64,
    generated: bool,
}

pub struct WorldStreamer {
    block_registry: Arc<BlockRegistry>,
    renderer: Arc<Mutex<Renderer>>,
    cluster_mat_pipeline: Arc<MaterialPipeline>,
    // in meters
    render_distance: u32,
    stream_pos: DVec3,
    clusters: [HashMap<I32Vec3, WorldCluster>; MAX_LOD + 1], // [LOD] -> clusters
}

pub trait ClusterProvider {}

impl WorldStreamer {
    const MIN_RENDER_DISTANCE: u32 = 128;
    const MAX_RENDER_DISTANCE: u32 = 65536;

    pub fn set_stream_pos(&mut self, pos: DVec3) {
        self.stream_pos = pos;
    }

    pub fn set_render_distance(&mut self, render_distance: u32) {
        self.render_distance = render_distance
            .min(Self::MAX_RENDER_DISTANCE)
            .max(Self::MIN_RENDER_DISTANCE);
    }

    fn cluster_size(level: u32) -> u32 {
        2_u32.pow(6 + level)
    }

    fn cluster_aligned_pos(pos: DVec3, cluster_lod: u32) -> I32Vec3 {
        let cluster_step_size = Self::cluster_size(cluster_lod) as f64;
        glm::try_convert(glm::floor(&(pos / cluster_step_size))).unwrap()
    }

    fn handle_cluster_side_updates(&self, level: usize, pos: &I32Vec3) {
        let mut neighbours = SmallVec::<[(usize, I32Vec3); 24]>::new();

        let cluster_size1 = Self::cluster_size(level as u32) as i32;
        let cluster_size2 = Self::cluster_size(level as u32 + 1) as i32;

        // Lower level
        if level > 0 {
            fn map_dst_pos(d: &I32Vec3, k: i32, l: i32) -> I32Vec3 {
                let dx = (d.x != 0) as i32;
                let dy = (d.y != 0) as i32;
                let dz = (d.z != 0) as i32;

                let (k, l) = (k + 1, l + 1);
                let x = -1 + k * dy + k * dz + 3 * (d.x > 0) as i32;
                let y = -1 + k * dx + l * dz + 3 * (d.y > 0) as i32;
                let z = -1 + l * dx + l * dy + 3 * (d.z > 0) as i32;

                I32Vec3::new(x, y, z)
            }

            let cluster_size0 = Self::cluster_size(level as u32 - 1) as i32;

            for d in &Facing::DIRECTIONS {
                for k in 0..2 {
                    for l in 0..2 {
                        let pos2 = pos + map_dst_pos(d, k, l) * cluster_size0;

                        if self.clusters[level - 1].contains_key(&pos2) {
                            neighbours.push((level - 1, pos2));
                        }
                    }
                }
            }
        }

        // Current level
        {
            for d in &Facing::DIRECTIONS {
                let pos2 = pos + d * cluster_size1;

                if self.clusters[level].contains_key(&pos2) {
                    neighbours.push((level, pos2));
                }
            }
        }

        dbg!(pos);

        // Higher level
        if level < MAX_LOD {
            for d in &Facing::DIRECTIONS {
                let pos2 = pos
                    + d * cluster_size2
                    + d.zip_map(pos, |d, p| {
                        (-cluster_size1 * (d > 0) as i32) - (p % cluster_size2 * (d == 0) as i32).abs()
                    });

                // 192 0 -128

                if *pos == I32Vec3::new(64 * 3, 0, 0) {
                    // dbg!(d, pos2 - pos);
                }

                if glm::all(&pos2.map(|v| v % cluster_size2 != 0)) {
                    continue;
                }

                if self.clusters[level + 1].contains_key(&pos2) {
                    dbg!(d, pos2);
                    neighbours.push((level + 1, pos2));
                }
            }
        }

        let cluster = &self.clusters[level][pos];
        let mut cluster = cluster.cluster.lock().unwrap();

        for (side_level, side_pos) in neighbours {
            let side_cluster = &self.clusters[side_level][&side_pos];
            let mut side_cluster = side_cluster.cluster.lock().unwrap();
            let offset = side_pos - pos;

            cluster.paste_outer_side_occlusion(&mut side_cluster, offset);
        }
    }

    pub fn calc_cluster_layout(&self) -> Vec<HashSet<I32Vec3>> {
        const R: i32 = (LOD0_RANGE / cluster::SIZE) as i32;
        const D: i32 = R * 2 + 1;

        let mut cluster_layout = vec![HashSet::with_capacity(512); MAX_LOD + 1];
        let mut masks = [[[[false; D as usize]; D as usize]; D as usize]; MAX_LOD + 2];

        for i in 0..(MAX_LOD + 1) {
            let cluster_size = Self::cluster_size(i as u32) as i32;
            let stream_pos_i = Self::cluster_aligned_pos(self.stream_pos, i as u32);
            let r = (R * cluster_size) as f64;

            // Fill mask of used clusters
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        let pos = stream_pos_i + I32Vec3::new(x, y, z).add_scalar(-R);
                        let center = (pos * cluster_size).add_scalar(cluster_size / 2);

                        let dist = glm::distance(&self.stream_pos, &glm::convert(center));

                        if dist <= r && dist <= (self.render_distance as f64) {
                            let p = [(R + x) as usize / 2, (R + y) as usize / 2, (R + z) as usize / 2];
                            masks[i + 1][p[0]][p[1]][p[2]] = true;
                        }
                    }
                }
            }

            let mask0 = &masks[i];
            let mask1 = &masks[i + 1];

            // Calculate new cluster positions
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        if !mask1[x as usize][y as usize][z as usize] {
                            continue;
                        }

                        let pos = stream_pos_i + I32Vec3::new(x, y, z).add_scalar(-R) * 2;
                        let in_p = [x * 2 - R, y * 2 - R, z * 2 - R];

                        if (in_p[0] < 0 || in_p[0] >= D)
                            || (in_p[1] < 0 || in_p[1] >= D)
                            || (in_p[2] < 0 || in_p[2] >= D)
                        {
                            continue;
                        }

                        for x2 in 0..2_usize {
                            for y2 in 0..2_usize {
                                for z2 in 0..2_usize {
                                    let p = [
                                        in_p[0] as usize + x2,
                                        in_p[1] as usize + y2,
                                        in_p[2] as usize + z2,
                                    ];
                                    if p[0] < D as usize
                                        && p[1] < D as usize
                                        && p[2] < D as usize
                                        && mask0[p[0]][p[1]][p[2]]
                                    {
                                        continue;
                                    }

                                    let pos = pos + I32Vec3::new(x2 as i32, y2 as i32, z2 as i32);
                                    cluster_layout[i].insert(pos);
                                }
                            }
                        }
                    }
                }
            }
        }

        cluster_layout
    }

    pub fn on_update(&mut self) {
        // Add/remove clusters
        {
            let cluster_layout = self.calc_cluster_layout();

            let renderer = self.renderer.lock().unwrap();
            let device = renderer.device().clone();
            let scene = renderer.scene();
            let transform_comps = scene.storage::<component::Transform>();

            let curr_time_secs = Instant::now().elapsed().as_secs();

            for i in 0..(MAX_LOD + 1) {
                // Remove unnecessary clusters
                let mut entities_to_remove = Vec::with_capacity(512);

                self.clusters[i].retain(|pos, world_cluster| {
                    if (curr_time_secs - world_cluster.creation_time_secs) < 5
                        || cluster_layout[i].contains(pos)
                    {
                        true
                    } else {
                        entities_to_remove.push(world_cluster.entity);
                        false
                    }
                });
                scene.remove_entities(&entities_to_remove);

                // -----------------------------------------------------------------------------------

                let mut scene_entities = scene.entities().lock().unwrap();
                let mut transform_comps = transform_comps.write().unwrap();

                // Add missing clusters
                for pos in &cluster_layout[i] {
                    let node_size = 2_u32.pow(i as u32);
                    let pos = pos * (cluster::SIZE as i32) * (node_size as i32);

                    if let hash_map::Entry::Vacant(entry) = self.clusters[i].entry(pos) {
                        let cluster = cluster::new(&self.block_registry, &device, node_size);

                        let transform_comp = component::Transform::new(
                            Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                            Vec3::new(0.0, 0.0, 0.0),
                            Vec3::new(1.0, 1.0, 1.0),
                        );

                        let entity = scene_entities.create();
                        transform_comps.set(entity, transform_comp);

                        entry.insert(WorldCluster {
                            cluster: Arc::new(Mutex::new(cluster)),
                            entity,
                            creation_time_secs: curr_time_secs,
                            generated: false,
                        });
                    }
                }
            }
        }

        // Generate clusters
        {
            for (i, level) in self.clusters.iter_mut().enumerate() {
                level.par_iter_mut().for_each(|(pos, world_cluster)| {
                    if pos.y != 0 {
                        return;
                    }

                    let node_size = 2_u32.pow(i as u32);

                    let mut cluster = world_cluster.cluster.lock().unwrap();
                    generator::generate_cluster(&mut cluster, *pos, node_size);
                    // cluster.set_densities(&points);
                });
            }
        }

        // Generate meshes
        {
            let renderer = self.renderer.lock().unwrap();
            let scene = renderer.scene();

            let entities = scene.entities();
            let transform_comps = scene.storage::<component::Transform>();
            let renderer_comps = scene.storage::<component::Renderer>();
            let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
            let children_comps = scene.storage::<component::Children>();

            let mut entities = entities.lock().unwrap();
            let transform_comps = transform_comps.write().unwrap();
            let renderer_comps = renderer_comps.write().unwrap();
            let vertex_mesh_comps = vertex_mesh_comps.write().unwrap();
            let children_comps = children_comps.write().unwrap();

            let mut d = cluster::UpdateSystemData {
                mat_pipeline: Arc::clone(&self.cluster_mat_pipeline),
                entities: &mut entities,
                transform: transform_comps,
                renderer: renderer_comps,
                vertex_mesh: vertex_mesh_comps,
                children: children_comps,
            };

            for (i, level) in self.clusters.iter().enumerate() {
                level.iter().for_each(|(pos, world_cluster)| {
                    self.handle_cluster_side_updates(i, pos);
                });
                level.par_iter().for_each(|(pos, world_cluster)| {
                    let mut cluster = world_cluster.cluster.lock().unwrap();
                    // let fake_seam = cluster::Seam::new(cluster.node_size());
                    // let seam = self.create_seam_for_cluster(i, pos);
                    // let seam = world_cluster.seam.as_ref().unwrap_or(&fake_seam);
                    // cluster.fill_seam_densities(seam);
                    cluster.update_mesh(0.75);
                    // let seam = world_cluster.seam.as_ref().unwrap_or(&fake_seam);
                });
                level.iter().for_each(|(_, world_cluster)| {
                    // let raw_vertex_mesh = component::VertexMesh::new(
                    //     &world_cluster.cluster.lock().unwrap().vertex_mesh().raw(),
                    // );
                    world_cluster
                        .cluster
                        .lock()
                        .unwrap()
                        .update_renderable(world_cluster.entity, &mut d);
                    // *vertex_mesh_comps.get_mut(world_cluster.entity).unwrap() = raw_vertex_mesh;
                });
            }
            /*for (i, level) in self.clusters.iter().enumerate() {
                level.par_iter().for_each(|(pos, world_cluster)| {
                    let mut cluster = world_cluster.cluster.lock().unwrap();
                    let seam = self.create_seam_for_cluster(i, pos);
                    cluster.fill_seam_densities(&seam);
                    cluster.update_mesh(&seam, 0.0);
                });
            }*/
        }
    }
}

pub fn new(
    block_registry: &Arc<BlockRegistry>,
    renderer: &Arc<Mutex<Renderer>>,
    cluster_mat_pipeline: &Arc<MaterialPipeline>,
) -> WorldStreamer {
    WorldStreamer {
        block_registry: Arc::clone(block_registry),
        renderer: Arc::clone(renderer),
        cluster_mat_pipeline: Arc::clone(cluster_mat_pipeline),
        render_distance: 0,
        stream_pos: DVec3::new(0.0, 0.0, 0.0),
        clusters: Default::default(),
    }
}
