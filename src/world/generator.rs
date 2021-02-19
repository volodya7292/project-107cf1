use crate::object::cluster;
use nalgebra as na;
use nalgebra_glm as glm;
use simdnoise::NoiseBuilder;

fn normalize_densities(densities: &[f32], size: na::Vector3<u32>) -> Vec<f32> {
    let mut new_points = vec![0_f32; (size.x * size.y * size.z) as usize];

    let iso_value = cluster::ISO_VALUE_NORM;
    let index = |x: u32, y: u32, z: u32| -> usize { (x * size.y * size.z + y * size.z + z) as usize };

    for x in 0..size.x {
        for y in 0..size.y {
            for z in 0..size.z {
                let v = densities[index(x, y, z)];
                let v_xyz0 = na::Vector3::new(
                    densities[index(x.saturating_sub(1), y, z)],
                    densities[index(x, y.saturating_sub(1), z)],
                    densities[index(x, y, z.saturating_sub(1))],
                );
                let v_xyz1 = na::Vector3::new(
                    densities[index((x + 1).min(size.x - 1), y, z)],
                    densities[index(x, (y + 1).min(size.y - 1), z)],
                    densities[index(x, y, (z + 1).min(size.z - 1))],
                );

                let a_d = v - iso_value;
                let bce_d = v_xyz0.add_scalar(-iso_value);
                let bce_d2 = v_xyz1.add_scalar(-iso_value);

                let a_t = ((v >= iso_value) as i8 as f32 - a_d) / a_d;
                let bce_t = (v_xyz0.map(|e| (e >= iso_value) as i8 as f32) - v_xyz0).component_div(&bce_d);
                let bce_t2 = (v_xyz1.map(|e| (e >= iso_value) as i8 as f32) - v_xyz1).component_div(&bce_d2);

                let t = a_t.min(bce_t.min()).min(bce_t2.min());
                let n_v = v + a_d * t;

                new_points[index(x, y, z)] = n_v;
            }
        }
    }

    new_points
}

pub fn generate_cluster(pos: na::Vector3<i32>, node_size: u32) -> Vec<cluster::DensityPointInfo> {
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

    let sample_noise =
        |x, y, z| -> f32 { noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0 };

    let index = |x: usize, y: usize, z: usize| -> usize {
        (x * cluster::SIZE * cluster::SIZE + y * cluster::SIZE + z) as usize
    };

    let mut densities = vec![0_f32; cluster::VOLUME];
    let mat = if pos.x > 0 { 0 } else { 1 };

    for x in 0..(cluster::SIZE) {
        for y in 0..(cluster::SIZE) {
            for z in 0..(cluster::SIZE) {
                let n_v = sample_noise(x, y, z);

                /*

                vec2 coord = fragCoord / 2000.0;

                float c = f(coord);

                c += 0.9;
                c *= (fragCoord.y / 450.0);

                if (c > 0.9) {
                    //c = 1.0;
                } else {
                    //c = 0.0;
                }

                fragColor = vec4(c, c, c, 1.0);


                 */

                // let n_v = ((n_v as f32 + (64 - (pos.y + y as i32) * (node_size as i32)) as f32 / 10.0) / 2.0)
                //     .max(0.0)
                //     .min(1.0);

                let p0 = na::Vector3::new(pos.x as f32 + x as f32, 0.0, pos.z as f32 + z as f32);
                let d = na::Vector3::new(pos.x as f32 + x as f32, y as f32, pos.z as f32 + z as f32);
                let h = (((d - p0).magnitude() * node_size as f32 + 32.0) / 64.0).clamp(0.0, 1.0);

                // let h = ((y as f32 * node_size as f32 + 32.0) / 65.0).clamp(0.0, 1.0);

                let n_v = (n_v + 2.0) * (1.0 - h);

                densities[index(x, y, z)] = n_v.clamp(0.0, 1.0);
            }
        }
    }

    normalize_densities(&densities, na::convert(na::Vector3::from_element(cluster::SIZE)));

    let mut points = Vec::<cluster::DensityPointInfo>::with_capacity(cluster::VOLUME);

    for x in 0..(cluster::SIZE) {
        for y in 0..(cluster::SIZE) {
            for z in 0..(cluster::SIZE) {
                points.push(cluster::DensityPointInfo {
                    pos: [x as u8, y as u8, z as u8, 0],
                    point: cluster::DensityPoint {
                        density: (densities[index(x, y, z)] * 255.0) as u8,
                        material: mat,
                    },
                });
            }
        }
    }

    points
}
