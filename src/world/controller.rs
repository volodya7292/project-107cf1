use crate::object::cluster;
use crate::object::cluster::{Cluster, ClusterAdjacency};
use crate::renderer::{component, Renderer};
use nalgebra as na;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use specs::WorldExt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const MAX_LOD: usize = 4;

#[derive(Clone)]
struct WorldCluster {
    cluster: Arc<Mutex<Cluster>>,
    entity: specs::Entity,
    creation_time_secs: u64,
    generated: bool,
}

pub struct WorldController {
    renderer: Arc<Mutex<Renderer>>,
    render_distance: u32, // in meters
    camera_pos_in_clusters: [i32; 3],
    clusters: [HashMap<[i32; 3], WorldCluster>; MAX_LOD + 1], // [LOD] -> clusters
}

impl WorldController {
    const MIN_RENDER_DISTANCE: u32 = 128;
    const MAX_RENDER_DISTANCE: u32 = 8192;

    pub fn set_render_distance(&mut self, render_distance: u32) {
        self.render_distance = render_distance
            .min(Self::MAX_RENDER_DISTANCE)
            .max(Self::MIN_RENDER_DISTANCE);
    }

    fn cluster_size(level: u32) -> u32 {
        2_u32.pow(6 + level)
    }

    fn set_cluster_boundaries(&self, lod_index: u32, cluster_pos: [i32; 3]) {
        if let Some(world_cluster) = self.clusters[lod_index as usize].get(&cluster_pos) {
            let mut renderer = self.renderer.lock().unwrap();
            let world = renderer.world_mut();
            let mut cluster_comp = world.write_component::<Cluster>();

            let mut adjacency = Box::new(ClusterAdjacency::default());
            adjacency.densities =
                Vec::with_capacity(cluster::SIZE * cluster::SIZE * 6 + cluster::SIZE * 12 + 8);

            macro_rules! side {
                ($name: ident, $cluster_pos: expr, $v0: ident, $v1: ident, $get_pos: expr) => {
                    if let Some(world_cluster) = self.clusters[lod_index as usize].get(&$cluster_pos) {
                        let cluster = cluster_comp.get(world_cluster.entity).unwrap();

                        for $v0 in 0..cluster::SIZE {
                            for $v1 in 0..cluster::SIZE {
                                let mut temp_layers = [Default::default(); cluster::MAX_CELL_LAYERS];
                                let count = cluster.get_density_layers(
                                    [$get_pos[0] as u8, $get_pos[1] as u8, $get_pos[2] as u8],
                                    &mut temp_layers,
                                );

                                adjacency.$name[$v0][$v1] =
                                    cluster::calc_density_index(adjacency.densities.len() as u32, count);
                                adjacency
                                    .densities
                                    .extend_from_slice(&temp_layers[0..(count as usize)])
                            }
                        }
                    }
                };
            }

            macro_rules! edge {
                ($name: ident, $cluster_pos: expr, $v: ident, $get_pos: expr) => {
                    if let Some(world_cluster) = self.clusters[lod_index as usize].get(&$cluster_pos) {
                        let cluster = cluster_comp.get(world_cluster.entity).unwrap();

                        for $v in 0..cluster::SIZE {
                            let mut temp_layers = [Default::default(); cluster::MAX_CELL_LAYERS];
                            let count = cluster.get_density_layers(
                                [$get_pos[0] as u8, $get_pos[1] as u8, $get_pos[2] as u8],
                                &mut temp_layers,
                            );

                            adjacency.$name[$v] =
                                cluster::calc_density_index(adjacency.densities.len() as u32, count);
                            adjacency
                                .densities
                                .extend_from_slice(&temp_layers[0..(count as usize)])
                        }
                    }
                };
            }

            macro_rules! corner {
                ($name: ident, $cluster_pos: expr, $get_pos: expr) => {
                    if let Some(world_cluster) = self.clusters[lod_index as usize].get(&$cluster_pos) {
                        let cluster = cluster_comp.get(world_cluster.entity).unwrap();

                        let mut temp_layers = [Default::default(); cluster::MAX_CELL_LAYERS];
                        let count = cluster.get_density_layers(
                            [$get_pos[0] as u8, $get_pos[1] as u8, $get_pos[2] as u8],
                            &mut temp_layers,
                        );

                        adjacency.$name =
                            cluster::calc_density_index(adjacency.densities.len() as u32, count);
                        adjacency
                            .densities
                            .extend_from_slice(&temp_layers[0..(count as usize)])
                    };
                };
            }

            // Sides
            // ---------------------------------------------------------------------------------------------------------

            side!(
                side_x0,
                [cluster_pos[0] - 1, cluster_pos[1], cluster_pos[2]],
                y,
                z,
                [cluster::SIZE - 1, y, z]
            );
            side!(
                side_x1,
                [cluster_pos[0] + 1, cluster_pos[1], cluster_pos[2]],
                y,
                z,
                [0, y, z]
            );
            side!(
                side_y0,
                [cluster_pos[0], cluster_pos[1] - 1, cluster_pos[2]],
                x,
                z,
                [x, cluster::SIZE - 1, z]
            );
            side!(
                side_y1,
                [cluster_pos[0], cluster_pos[1] + 1, cluster_pos[2]],
                x,
                z,
                [x, 0, z]
            );
            side!(
                side_z0,
                [cluster_pos[0], cluster_pos[1], cluster_pos[2] - 1],
                x,
                y,
                [x, y, cluster::SIZE - 1]
            );
            side!(
                side_z1,
                [cluster_pos[0], cluster_pos[1], cluster_pos[2] + 1],
                x,
                y,
                [x, y, 0]
            );

            // Edges
            // ---------------------------------------------------------------------------------------------------------

            edge!(
                edge_x0_y0,
                [cluster_pos[0] - 1, cluster_pos[1] - 1, cluster_pos[2]],
                z,
                [cluster::SIZE - 1, cluster::SIZE - 1, z]
            );
            edge!(
                edge_x1_y0,
                [cluster_pos[0] + 1, cluster_pos[1] - 1, cluster_pos[2]],
                z,
                [0, cluster::SIZE - 1, z]
            );
            edge!(
                edge_x0_y1,
                [cluster_pos[0] - 1, cluster_pos[1] + 1, cluster_pos[2]],
                z,
                [cluster::SIZE - 1, 0, z]
            );
            edge!(
                edge_x1_y1,
                [cluster_pos[0] + 1, cluster_pos[1] + 1, cluster_pos[2]],
                z,
                [0, 0, z]
            );

            // -----------------------------------------------------

            edge!(
                edge_x0_z0,
                [cluster_pos[0] - 1, cluster_pos[1], cluster_pos[2] - 1],
                y,
                [cluster::SIZE - 1, y, cluster::SIZE - 1]
            );
            edge!(
                edge_x1_z0,
                [cluster_pos[0] + 1, cluster_pos[1], cluster_pos[2] - 1],
                y,
                [0, y, cluster::SIZE - 1]
            );
            edge!(
                edge_x0_z1,
                [cluster_pos[0] - 1, cluster_pos[1], cluster_pos[2] + 1],
                y,
                [cluster::SIZE - 1, y, 0]
            );
            edge!(
                edge_x1_z1,
                [cluster_pos[0] + 1, cluster_pos[1], cluster_pos[2] + 1],
                y,
                [0, y, 0]
            );

            // -----------------------------------------------------

            edge!(
                edge_y0_z0,
                [cluster_pos[0], cluster_pos[1] - 1, cluster_pos[2] - 1],
                x,
                [x, cluster::SIZE - 1, cluster::SIZE - 1]
            );
            edge!(
                edge_y1_z0,
                [cluster_pos[0], cluster_pos[1] + 1, cluster_pos[2] - 1],
                x,
                [x, 0, cluster::SIZE - 1]
            );
            edge!(
                edge_y0_z1,
                [cluster_pos[0], cluster_pos[1] - 1, cluster_pos[2] + 1],
                x,
                [x, cluster::SIZE - 1, 0]
            );
            edge!(
                edge_y1_z1,
                [cluster_pos[0], cluster_pos[1] + 1, cluster_pos[2] + 1],
                x,
                [x, 0, 0]
            );

            // Corners
            // ---------------------------------------------------------------------------------------------------------

            corner!(
                corner_x0_y0_z0,
                [cluster_pos[0] - 1, cluster_pos[1] - 1, cluster_pos[2] - 1],
                [cluster::SIZE - 1, cluster::SIZE - 1, cluster::SIZE - 1]
            );
            corner!(
                corner_x1_y0_z0,
                [cluster_pos[0] + 1, cluster_pos[1] - 1, cluster_pos[2] - 1],
                [0, cluster::SIZE - 1, cluster::SIZE - 1]
            );
            corner!(
                corner_x0_y1_z0,
                [cluster_pos[0] - 1, cluster_pos[1] + 1, cluster_pos[2] - 1],
                [cluster::SIZE - 1, 0, cluster::SIZE - 1]
            );
            corner!(
                corner_x1_y1_z0,
                [cluster_pos[0] + 1, cluster_pos[1] + 1, cluster_pos[2] - 1],
                [0, 0, cluster::SIZE - 1]
            );
            corner!(
                corner_x0_y0_z1,
                [cluster_pos[0] - 1, cluster_pos[1] - 1, cluster_pos[2] + 1],
                [cluster::SIZE - 1, cluster::SIZE - 1, 0]
            );
            corner!(
                corner_x1_y0_z1,
                [cluster_pos[0] + 1, cluster_pos[1] - 1, cluster_pos[2] + 1],
                [0, cluster::SIZE - 1, 0]
            );
            corner!(
                corner_x0_y1_z1,
                [cluster_pos[0] - 1, cluster_pos[1] + 1, cluster_pos[2] + 1],
                [cluster::SIZE - 1, 0, 0]
            );
            corner!(
                corner_x1_y1_z1,
                [cluster_pos[0] + 1, cluster_pos[1] + 1, cluster_pos[2] + 1],
                [0, 0, 0]
            );

            // Set adjacency
            let cluster = cluster_comp.get_mut(world_cluster.entity).unwrap();
            cluster.set_adjacent_densities(adjacency);
        }
    }

    pub fn on_update(&mut self) {
        // Adjust camera position & camera_pos_in_clusters
        {
            let renderer = self.renderer.lock().unwrap();
            let mut camera_comp = renderer.world().write_component::<component::Camera>();
            let camera = camera_comp.get_mut(renderer.get_active_camera()).unwrap();
            let pos = camera.position();

            let cluster_step_size = Self::cluster_size(0) as i32;
            let offset_in_clusters = [
                pos.x.floor() as i32 / cluster_step_size,
                pos.y.floor() as i32 / cluster_step_size,
                pos.z.floor() as i32 / cluster_step_size,
            ];
            let new_cam_pos = pos.map(|x| x % (cluster_step_size as f32));

            self.camera_pos_in_clusters = [
                self.camera_pos_in_clusters[0] + offset_in_clusters[0],
                self.camera_pos_in_clusters[1] + offset_in_clusters[1],
                self.camera_pos_in_clusters[2] + offset_in_clusters[2],
            ];
            camera.set_position(new_cam_pos);
        }

        // Remove out-of-render_distance clusters
        {
            let mut renderer = self.renderer.lock().unwrap();
            let world = renderer.world_mut();
            let cluster_step_size = Self::cluster_size(0) as i32;

            let cam_pos_v = na::Vector3::<i32>::new(
                self.camera_pos_in_clusters[0],
                self.camera_pos_in_clusters[1],
                self.camera_pos_in_clusters[2],
            );
            let render_distance = self.render_distance;
            let curr_time_secs = Instant::now().elapsed().as_secs();

            let mut entities_to_remove = Vec::with_capacity(65535);

            for level in &mut self.clusters {
                *level = level
                    .iter()
                    .filter_map(|(&pos, world_cluster)| {
                        let dist = na::Vector3::<f32>::new(
                            (pos[0] - cam_pos_v[0]) as f32,
                            (pos[1] - cam_pos_v[1]) as f32,
                            (pos[2] - cam_pos_v[2]) as f32,
                        )
                        .magnitude()
                            * (cluster_step_size as f32);

                        if (dist > render_distance as f32)
                            && (curr_time_secs - world_cluster.creation_time_secs) >= 5
                        {
                            // Remove cluster
                            entities_to_remove.push(world_cluster.entity);
                            None
                        } else {
                            Some((pos, world_cluster.clone()))
                        }
                    })
                    .collect();
            }

            world.delete_entities(&entities_to_remove).unwrap();
        }

        // Update clusters
        {
            for level in &self.clusters {
                level.par_iter().for_each(|(pos, world_cluster)| {
                    if let Ok(mut cluster) = world_cluster.cluster.try_lock() {
                        cluster.update_mesh(0.0);
                    }
                });
            }
        }
    }
}

pub fn new(renderer: &Arc<Mutex<Renderer>>) -> WorldController {
    WorldController {
        renderer: Arc::clone(renderer),
        render_distance: 0,
        camera_pos_in_clusters: Default::default(),
        clusters: Default::default(),
    }
}
