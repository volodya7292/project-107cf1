use std::sync::Arc;
use std::time::Instant;

use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, DVec3, DVec4, I64Vec2, I64Vec3, Vec3};
use noise::Seedable;
use rand::Rng;

use engine::utils::noise::HybridNoise;
use engine::utils::voronoi_noise::VoronoiNoise2D;
use engine::utils::white_noise::WhiteNoise;

use crate::core::overworld::generator::OverworldGenerator;
use crate::core::overworld::position::BlockPos;
use crate::core::overworld::structure::world::biome::MeanTemperature;
use crate::core::overworld::structure::world::WorldState;
use crate::core::registry::Registry;

pub fn make_world_prototype_image(generator: &OverworldGenerator) {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let land_density = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(0));

    let temp0 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(50));
    let temp1 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(51));
    let temp2 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(52));

    let moist0 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(100));
    let moist1 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(101));
    let moist2 = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(102));

    let n = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new(0));

    let p_noise = noise::SuperSimplex::new(0);
    let white_noise = WhiteNoise::new(0);
    let mut v = VoronoiNoise2D::new();

    // TODO: implement SIMD noise

    let b_freq = 30.0;

    let main_registry = generator.main_registry();
    let registry = main_registry.registry();
    let world_pos = generator.gen_world_pos(BlockPos(I64Vec3::zeros()));
    let world_seed = generator.get_world_seed(world_pos.center_pos);

    let world = WorldState::new(world_seed, registry);

    // let world = WorldState::new(0, registry);

    let mut n_min = f64::MAX;
    let mut n_max = f64::MIN;

    let mut process = |p: DVec2| -> DVec3 {
        // let land = land_ocean.sample(p, 10.0, 0.5) * 0.5 + 0.5;
        // let f = temperature.sample(p, 5.0, 0.5) * 0.5 + 0.5;
        // let noise = n.sample(p, 5.0, 0.5) * 0.5 + 0.5;
        // let noise = n.sample(p, (f * 5.0).clamp(0.001, 5.0), 0.5) * 0.5 + 0.5;

        // let t0 = temperature.sample(p, 100.0, 0.5);
        // let t1 = moisture.sample(p, 100.0, 0.5);

        // let b = w.get([p.x, p.y]) * 0.5 + 0.5;

        // let (b, pivot) = world.biome_2d_at(I64Vec2::new((p.x * 10000.0) as i64, (p.y * 10000.0) as i64));

        // let distort0 = (temperature.sample(p, b_freq, 0.5) * 2.0 - 1.0) * 0.1;
        // let distort1 = (moisture.sample(p, b_freq, 0.5) * 2.0 - 1.0) * 0.1;
        //
        // let (b_pos, voronoi_d) = v.sample(DVec2::new(p.x * b_freq, p.y * b_freq));

        // let distr = rand::distributions::WeightedIndex::new([1, 1, 1, 1, 1, 1, 1, 1, 4, 1]).unwrap();
        // let b = white_noise
        //     .state()
        //     .next(b_pos.x as u64)
        //     .next(b_pos.y as u64)
        //     .rng()
        //     .gen_range(0..10);

        // let col = white_noise.state().next(pivot.x).next(pivot.y).rng().gen::<f64>();

        // let a = ;

        // .gen::<f64>();

        // rand::distributions::Dis
        // rand_distr::

        // let siblings = v.sample_3x3(b_pos);

        // let b = p_noise.get([p.x * 50.0, p.y * 50.0]);

        // let noise = land;

        // let b = world.select_biome_idx(glm::convert_unchecked(p * 1000.0), 10);
        let biome = world.biome_2d_at(glm::convert_unchecked::<_, I64Vec2>(p * 4096.0).add_scalar(-2048));
        let n_biomes = registry.biomes().len() as u32;

        let t1 = temp0.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let t2 = temp1.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let t3 = temp2.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let t = (t1 + t2 + t3) / 3.0;
        // let t = world.sample_temperature(glm::convert_unchecked::<_, I64Vec2>(p * 2048.0))
        //     - (MeanTemperature::MIN as i32);
        // let t = t as f64 / 60.0;

        let land_col: DVec3 = match t {
            0.0..=0.3 => glm::vec3(0.3, 0.3, 0.3),
            0.3..=0.43 => glm::vec3(0.4, 0.8, 0.4),
            0.43..=0.57 => glm::vec3(0.0, 1.0, 0.0),
            0.57..=0.7 => glm::vec3(1.0, 1.0, 0.0),
            0.7..=1.0 => glm::vec3(1.0, 0.0, 0.0),
            _ => glm::vec3(t, t, t),
        };

        let m1 = moist0.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let m2 = moist1.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let m3 = moist2.sample(p * 10.0, 0.5) * 0.5 + 0.5;
        let m = (m1 + m2 + m3) / 3.0;

        // let m = world.sample_humidity(glm::convert_unchecked::<_, I64Vec2>(p * 2048.0));
        // let m = m as f64 / 100.0;

        let m_col: DVec3 = match m {
            0.0..=0.3 => glm::vec3(0.0, 0.0, 1.0),
            0.3..=0.43 => glm::vec3(0.0, 0.5, 0.5),
            0.43..=0.57 => glm::vec3(0.0, 1.0, 0.0),
            0.57..=0.7 => glm::vec3(0.5, 0.5, 0.0),
            0.7..=1.0 => glm::vec3(1.0, 0.0, 0.0),
            _ => glm::vec3(m, m, m),
        };

        let rain_precip = DVec4::new(0.0, 0.0, 1.0, 0.4); // t > 0.5
        let snow_precip = DVec4::new(1.0, 1.0, 1.0, 1.0); // t < 0.5
        let mut precip = if t > 0.5 { rain_precip } else { snow_precip };

        // n_min = n_min.min(v);
        // n_max = n_max.max(v);

        // 0C - < 40%
        // -6C - < 100%

        let col = glm::mix(&land_col, &precip.xyz(), if t < 0.5 { 1.0 } else { precip.w * m });

        let d = 1.0_f64.min(glm::distance(&DVec2::new(0.5, 0.5), &p) * 2.0);
        let grad = glm::smoothstep(0.0, 1.0, 4.0 * (1.0 - d));

        // y = 0.4 - (0.1 - (x - 0.4)) / 0.1

        // let biome = b as f64 / 9.0;

        // (1.0 - voronoi_d.pow(3.0)) * grad
        // col * grad
        // (t_col + m_col) * 0.5
        // land_col
        // m_col
        // noise * grad
        DVec3::from_element(biome as f64 / (n_biomes as f64 - 1.0))
    };

    let t0 = Instant::now();

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            // buf[i] = (v * 255.0) as u8;
            // buf[i + 1] = buf[i];
            // buf[i + 2] = buf[i];

            buf[i] = (v.x * 255.0) as u8;
            buf[i + 1] = (v.y * 255.0) as u8;
            buf[i + 2] = (v.z * 255.0) as u8;
        }
    }

    dbg!(n_min, n_max);

    let t1 = Instant::now();

    println!(
        "T: {}",
        (t1 - t0).as_secs_f64() / 1024.0 / 1024.0 * 24.0_f64.powi(3)
    );

    image::save_buffer("noise_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}

pub fn make_climate_graph_image(registry: &Arc<Registry>) {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let world = WorldState::new(0, registry);
    let white_noise = WhiteNoise::new(0);

    let mut process = |p: DVec2| -> Vec3 {
        let t = p.x * 60.0 - 30.;
        let h = (1.0 - p.y) * 100.0;
        let mut count = 1.0;

        let r = world
            .biomes_by_climate()
            .locate_all_at_point(&[t as f32, h as f32, 0.5])
            .map(|v| {
                let mut rng = white_noise.state().next(v.biome_id as u64).rng();
                Vec3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>())
            })
            .reduce(|accum, v| {
                count += 1.0;
                accum + v
            });

        r.unwrap_or(Vec3::from_element(0.0)) / count
        // let biome = registry.get_biome(r.biome_id).unwrap();
    };

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            // buf[i] = (v * 255.0) as u8;
            // buf[i + 1] = buf[i];
            // buf[i + 2] = buf[i];

            buf[i] = (v.x * 255.0) as u8;
            buf[i + 1] = (v.y * 255.0) as u8;
            buf[i + 2] = (v.z * 255.0) as u8;
        }
    }

    image::save_buffer("climate_graph_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}
