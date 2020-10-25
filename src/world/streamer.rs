use crate::object::cluster;
use crate::object::cluster::Cluster;
use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::{component, Renderer};
use nalgebra as na;
use nalgebra::SimdComplexField;
use nalgebra_glm as glm;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use simdnoise::NoiseBuilder;
use specs::{Builder, WorldExt};
use std::collections::{hash_map, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const MAX_LOD: usize = 4;
pub const LOD0_RANGE: usize = 128;

#[derive(Clone)]
struct WorldCluster {
    cluster: Arc<Mutex<Cluster>>,
    entity: specs::Entity,
    creation_time_secs: u64,
    generated: bool,
}

pub struct WorldStreamer {
    renderer: Arc<Mutex<Renderer>>,
    cluster_mat_pipeline: Arc<MaterialPipeline>,
    // in meters
    render_distance: u32,
    stream_pos: na::Vector3<f64>,
    clusters: [HashMap<na::Vector3<i32>, WorldCluster>; MAX_LOD + 1], // [LOD] -> clusters
}

pub trait ClusterProvider {}

impl WorldStreamer {
    const MIN_RENDER_DISTANCE: u32 = 128;
    const MAX_RENDER_DISTANCE: u32 = 65536;

    pub fn set_stream_pos(&mut self, pos: na::Vector3<f64>) {
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

    fn cluster_aligned_pos(pos: na::Vector3<f64>, cluster_lod: u32) -> na::Vector3<i32> {
        let cluster_step_size = Self::cluster_size(cluster_lod) as f64;
        na::try_convert(glm::floor(&(pos / cluster_step_size))).unwrap()
    }

    // TODO
    fn get_seam_clusters(&self) {
        unimplemented!()
    }

    /*pub fn on_update(&mut self) {
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
                        //cluster.update_mesh(0.0); TODO
                    }
                });
            }
        }

        // LOD -> radius in clusters (area)
        // ---------------
        // lod -> 2 ^ (1 + lod)
        // 0 -> 2 (13)
        // 1 -> 4 (51)
        // 2 -> 8 (202)
        // 3 -> 16 (805)
        // 4 -> 32 (3217)
        // 5 -> 64 (12868)
        // 6 -> 128 (51472)

        // AVG face count per cluster T = 4096

        // Average total triangle count
        // ---------------
        // circle_area(2) * T + (circle_area(4) - circle_area(2)) * (T * 4^-1) + (circle_area(8) - circle_area(4)) * (T * 4^-2)

        // 13 * 4096 + (51 - 13) * 1024 + (202 - 51) * 256 + (805 - 202) * 64 + (3217 - 805) * 16 + (12868 - 3217) * 4 + (51472 - 12868) * 1
        // = 285208
    }*/

    pub fn on_update(&mut self) {
        // Remove out-of-render_distance clusters

        // TODO: make corrections on distance calculation
        /*{
            let mut renderer = self.renderer.lock().unwrap();
            let world = renderer.world_mut();
            let cluster_step_size = Self::cluster_size(0) as f64;
            let render_distance = self.render_distance as f64;

            let stream_pos_i: na::Vector3<i32> =
                na::try_convert(glm::floor(&(self.stream_pos / cluster_step_size))).unwrap();
            let curr_time_secs = Instant::now().elapsed().as_secs();

            let mut entities_to_remove = Vec::with_capacity(65535);

            for level in &mut self.clusters {
                level.retain(|&pos, world_cluster| {
                    let dist = na::convert::<na::Vector3<i32>, na::Vector3<f64>>(pos - stream_pos_i)
                        .magnitude()
                        * cluster_step_size;

                    if (dist > render_distance) && (curr_time_secs - world_cluster.creation_time_secs) > 5 {
                        entities_to_remove.push(world_cluster.entity);
                        false
                    } else {
                        true
                    }
                });
            }

            world.delete_entities(&entities_to_remove).unwrap();
        }*/

        // Add new clusters
        {
            let mut to_add = Vec::with_capacity(1024);

            let mut renderer = self.renderer.lock().unwrap();
            let device = Arc::clone(renderer.device());
            let world = renderer.world_mut();

            const R: i32 = (LOD0_RANGE / cluster::SIZE) as i32;
            const D: i32 = R * 2 + 1;
            const R2: usize = (R / 2) as usize;
            const D2: usize = R2 * 2 + 1;

            let mut masks = [[[[false; D as usize]; D as usize]; D as usize]; MAX_LOD + 2];

            for (i, level) in self.clusters.iter_mut().enumerate() {
                let cluster_size = Self::cluster_size(i as u32) as i32;
                let stream_pos_i = Self::cluster_aligned_pos(self.stream_pos, i as u32);
                let r = (R * cluster_size) as f64;

                for x in 0..D {
                    for y in 0..D {
                        for z in 0..D {
                            let pos = stream_pos_i + na::Vector3::new(x, y, z).add_scalar(-R);
                            let center = (pos * cluster_size).add_scalar(cluster_size / 2);

                            let dist = glm::distance(&self.stream_pos, &na::convert(center));

                            if dist <= r && dist <= (self.render_distance as f64) {
                                let p = [(R + x) as usize / 2, (R + y) as usize / 2, (R + z) as usize / 2];
                                masks[i + 1][p[0]][p[1]][p[2]] = true;
                            }
                        }
                    }
                }

                let mask0 = &masks[i];
                let mask1 = &masks[i + 1];

                for x in 0..D {
                    for y in 0..D {
                        for z in 0..D {
                            if !mask1[x as usize][y as usize][z as usize] {
                                continue;
                            }

                            let pos = stream_pos_i + na::Vector3::new(x, y, z).add_scalar(-R) * 2;
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

                                        let pos = pos + na::Vector3::new(x2 as i32, y2 as i32, z2 as i32);

                                        if !level.contains_key(&pos) {
                                            to_add.push((i, pos));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let now = Instant::now().elapsed().as_secs();

            for (i, pos) in to_add {
                let node_size = 2_u32.pow(i as u32);
                let pos = pos * (cluster::SIZE as i32) * (node_size as i32);
                let cluster = cluster::new(&device, node_size);

                let transform_comp = component::Transform::new(
                    na::Vector3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                    na::Vector3::new(0.0, 0.0, 0.0),
                    na::Vector3::new(1.0, 1.0, 1.0),
                );
                let renderer_comp = component::Renderer::new(&device, &self.cluster_mat_pipeline, false);
                let mesh_comp = component::VertexMeshRef::new(&cluster.vertex_mesh().raw());

                let entity = world
                    .create_entity()
                    .with(transform_comp)
                    .with(renderer_comp)
                    .with(mesh_comp)
                    .build();

                self.clusters[i].insert(
                    pos,
                    WorldCluster {
                        cluster: Arc::new(Mutex::new(cluster)),
                        entity,
                        creation_time_secs: now,
                        generated: false,
                    },
                );
            }
        }

        // Generate clusters
        {
            for (i, level) in self.clusters.iter().enumerate() {
                level.par_iter().for_each(|(pos, world_cluster)| {
                    if pos.y != 0 {
                        return;
                    }

                    let node_size = 2_u32.pow(i as u32);

                    let noise = NoiseBuilder::gradient_3d_offset(
                        pos.x as f32 / (node_size as f32),
                        cluster::SIZE,
                        pos.y as f32 / (node_size as f32),
                        cluster::SIZE,
                        pos.z as f32 / (node_size as f32),
                        cluster::SIZE,
                    )
                    .with_seed(0)
                    .with_freq(1.0 / 50.0 * node_size as f32)
                    .generate();

                    let sample_noise = |x, y, z| -> f32 {
                        noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0
                    };

                    let mut points = Vec::<cluster::DensityPointInfo>::with_capacity(
                        cluster::SIZE * cluster::SIZE * cluster::SIZE,
                    );

                    for x in 0..(cluster::SIZE) {
                        for y in 0..(cluster::SIZE) {
                            for z in 0..(cluster::SIZE) {
                                let n_v = sample_noise(x, y, z);

                                let n_v = ((n_v as f32
                                    + (64 - (pos.y + y as i32) * (node_size as i32)) as f32 / 10.0)
                                    / 2.0)
                                    .max(0.0)
                                    .min(1.0);

                                points.push(cluster::DensityPointInfo {
                                    pos: [x as u8, y as u8, z as u8, 0],
                                    point: cluster::DensityPoint {
                                        density: (n_v * 255.0) as u8,
                                        material: 0,
                                    },
                                });
                            }
                        }
                    }

                    let mut cluster = world_cluster.cluster.lock().unwrap();

                    cluster.set_densities(&points);
                });
            }
        }

        // TODO
        // Generate seams
        {
            for (i, level) in self.clusters.iter().enumerate() {
                let node_size = 2_u32.pow(i as u32);

                level.par_iter().for_each(|(pos, world_cluster)| {
                    //for (pos, world_cluster) in level {
                    let mut neighbours = Vec::with_capacity(64);

                    let cluster_size1 = Self::cluster_size(i as u32) as i32;
                    let cluster_size2 = Self::cluster_size(i as u32 + 1) as i32;

                    // Lower level
                    if i > 0 {
                        let cluster_size0 = Self::cluster_size(i.saturating_sub(1) as u32) as i32;

                        for x in 0..3 {
                            for y in 0..3 {
                                for z in 0..3 {
                                    if x < 2 && y < 2 && z < 2 {
                                        continue;
                                    }
                                    let pos2 = pos + na::Vector3::new(x, y, z) * cluster_size0;

                                    if self.clusters[i - 1].contains_key(&pos2) {
                                        neighbours.push((i - 1, pos2));
                                    }
                                }
                            }
                        }
                    }

                    // Current level
                    {
                        for x in 0..2 {
                            for y in 0..2 {
                                for z in 0..2 {
                                    if x == 0 && y == 0 && z == 0 {
                                        continue;
                                    }
                                    let pos2 = pos + na::Vector3::new(x, y, z) * cluster_size1;

                                    if level.contains_key(&pos2) {
                                        neighbours.push((i, pos2));
                                    }
                                }
                            }
                        }
                    }

                    // Higher level
                    if i < MAX_LOD
                        && (pos.x + cluster_size1) % cluster_size2 == 0
                        && (pos.y + cluster_size1) % cluster_size2 == 0
                        && (pos.z + cluster_size1) % cluster_size2 == 0
                    {
                        for x in 0..2 {
                            for y in 0..2 {
                                for z in 0..2 {
                                    if x == 0 && y == 0 && z == 0 {
                                        continue;
                                    }
                                    let pos2 = pos.add_scalar(-cluster_size1)
                                        + na::Vector3::new(x, y, z) * cluster_size2;

                                    if self.clusters[i + 1].contains_key(&pos2) {
                                        neighbours.push((i + 1, pos2));
                                    }
                                }
                            }
                        }
                    }

                    let world_cluster = self.clusters[i].get(&pos).unwrap();
                    let mut cluster = world_cluster.cluster.lock().unwrap();

                    let mut seam = cluster::Seam::new(node_size);

                    for (j, neighbour_pos) in neighbours {
                        let offset =
                            neighbour_pos / (Self::cluster_size(j as u32) as i32) - pos / cluster_size1; // FIXME

                        let neighbour_world_cluster = &self.clusters[j][&neighbour_pos];
                        let mut neighbour_cluster = neighbour_world_cluster.cluster.lock().unwrap();

                        seam.insert(&mut neighbour_cluster, offset);
                    }

                    cluster.update_mesh(&seam, 0.0);
                    //}
                });
            }
        }
    }
}

pub fn new(renderer: &Arc<Mutex<Renderer>>, cluster_mat_pipeline: &Arc<MaterialPipeline>) -> WorldStreamer {
    WorldStreamer {
        renderer: Arc::clone(renderer),
        cluster_mat_pipeline: Arc::clone(cluster_mat_pipeline),
        render_distance: 0,
        stream_pos: na::Vector3::new(0.0, 0.0, 0.0),
        clusters: Default::default(),
    }
}
