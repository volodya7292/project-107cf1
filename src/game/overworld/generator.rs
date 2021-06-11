use crate::game::overworld::block::BlockProps;
use crate::game::overworld::cluster;
use crate::game::overworld::cluster::Cluster;
use nalgebra as na;
use nalgebra_glm as glm;
use nalgebra_glm::U32Vec3;
use simdnoise::NoiseBuilder;

pub fn generate_cluster(cluster: &mut Cluster, pos: na::Vector3<i32>, node_size: u32) {
    for x in 0..cluster::SIZE {
        for y in 0..cluster::SIZE {
            for z in 0..cluster::SIZE {
                let pos = U32Vec3::new(x as u32, y as u32, z as u32);
                cluster.set_block(pos, 0).props(BlockProps::default()).build();
            }
        }
    }

    // let noise = NoiseBuilder::gradient_3d_offset(
    //     pos.x as f32 / (node_size as f32),
    //     cluster::SIZE,
    //     pos.y as f32 / (node_size as f32),
    //     cluster::SIZE,
    //     pos.z as f32 / (node_size as f32),
    //     cluster::SIZE,
    // )
    // .with_seed(0)
    // .with_freq(1.0 / 50.0 * node_size as f32)
    // .generate();
    //
    // let sample_noise =
    //     |x, y, z| -> f32 { noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0 };
    //
    // let index = |x: usize, y: usize, z: usize| -> usize {
    //     (x * cluster::SIZE * cluster::SIZE + y * cluster::SIZE + z) as usize
    // };
    //
    // let mut densities = vec![0_f32; cluster::VOLUME];
    // let mat = if pos.x > 0 { 0 } else { 1 };
    //
    // for x in 0..(cluster::SIZE) {
    //     for y in 0..(cluster::SIZE) {
    //         for z in 0..(cluster::SIZE) {
    //             let n_v = sample_noise(x, y, z);
    //
    //             /*
    //
    //             vec2 coord = fragCoord / 2000.0;
    //
    //             float c = f(coord);
    //
    //             c += 0.9;
    //             c *= (fragCoord.y / 450.0);
    //
    //             if (c > 0.9) {
    //                 //c = 1.0;
    //             } else {
    //                 //c = 0.0;
    //             }
    //
    //             fragColor = vec4(c, c, c, 1.0);
    //
    //
    //              */
    //
    //             // let n_v = ((n_v as f32 + (64 - (pos.y + y as i32) * (node_size as i32)) as f32 / 10.0) / 2.0)
    //             //     .max(0.0)
    //             //     .min(1.0);
    //
    //             let p0 = na::Vector3::new(0.0, 0.0, 0.0);
    //             let d = na::Vector3::new(pos.x as f32 + x as f32, y as f32, pos.z as f32 + z as f32);
    //             let h = (((d - p0).magnitude() * node_size as f32 + 32.0) / 64.0).clamp(0.0, 1.0);
    //
    //             // let h = ((y as f32 * node_size as f32 + 32.0) / 65.0).clamp(0.0, 1.0);
    //
    //             let n_v = (n_v + 2.0) * (1.0 - h);
    //
    //             densities[index(x, y, z)] = n_v.clamp(0.0, 1.0);
    //         }
    //     }
    // }
    //
    // normalize_densities(&densities, na::convert(na::Vector3::from_element(cluster::SIZE)));
    //
    // let mut points = Vec::<cluster::DensityPointInfo>::with_capacity(cluster::VOLUME);
    //
    // for x in 0..(cluster::SIZE) {
    //     for y in 0..(cluster::SIZE) {
    //         for z in 0..(cluster::SIZE) {
    //             points.push(cluster::DensityPointInfo {
    //                 pos: [x as u8, y as u8, z as u8, 0],
    //                 point: cluster::DensityPoint {
    //                     density: (densities[index(x, y, z)] * 255.0) as u8,
    //                     material: mat,
    //                 },
    //             });
    //         }
    //     }
    // }
    //
    // points
}
