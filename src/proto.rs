use crate::game::overworld::structure::world::World;
use crate::game::registry::Registry;
use engine::utils::noise::HybridNoise;
use engine::utils::voronoi_noise::VoronoiNoise2D;
use engine::utils::white_noise::WhiteNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, I64Vec3};
use noise::Seedable;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

pub fn make_world_prototype_image(registry: &Arc<Registry>) {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let land_ocean = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new().set_seed(0));
    let temperature = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new().set_seed(50));
    let moisture = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new().set_seed(100));

    let n = HybridNoise::<2, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new().set_seed(0));

    let p_noise = noise::SuperSimplex::new();
    // let w = noise::Worley::new().set_frequency(10.0).set_displacement(1.0);
    let white_noise = WhiteNoise::new(0);
    let mut v = VoronoiNoise2D::new();

    // TODO: implement SIMD noise

    let b_freq = 30.0;

    let world = World::new(0, Arc::clone(registry));

    let mut process = |p: DVec2| -> f64 {
        // let land = land_ocean.sample(p, 10.0, 0.5) * 0.5 + 0.5;
        // let f = temperature.sample(p, 5.0, 0.5) * 0.5 + 0.5;
        // let noise = n.sample(p, 5.0, 0.5) * 0.5 + 0.5;
        // let noise = n.sample(p, (f * 5.0).clamp(0.001, 5.0), 0.5) * 0.5 + 0.5;

        // let t0 = temperature.sample(p, 100.0, 0.5);
        // let t1 = moisture.sample(p, 100.0, 0.5);

        // let b = w.get([p.x, p.y]) * 0.5 + 0.5;

        let (b, pivot) = world.biome_at(I64Vec3::new((p.x * 10000.0) as i64, 0, (p.y * 10000.0) as i64));

        // let distort0 = (temperature.sample(p, b_freq, 0.5) * 2.0 - 1.0) * 0.1;
        // let distort1 = (moisture.sample(p, b_freq, 0.5) * 2.0 - 1.0) * 0.1;
        //
        // let (b_pos, voronoi_d) = v.sample(DVec3::new(p.x * b_freq + distort0, p.y * b_freq + distort1, 0.0));

        // let distr = rand::distributions::WeightedIndex::new([1, 1, 1, 1, 1, 1, 1, 1, 4, 1]).unwrap();
        // let b = white_noise
        //     .state()
        //     .next(b_pos.x as u64)
        //     .next(b_pos.y as u64)
        //     .next(b_pos.z as u64)
        //     .rng()
        //     .sample::<usize, _>(distr);

        let col = white_noise.state().next(pivot.x).next(pivot.y).rng().gen::<f64>();

        // let a = ;

        // .gen::<f64>();

        // rand::distributions::Dis
        // rand_distr::

        // let siblings = v.sample_3x3(b_pos);

        // let b = p_noise.get([p.x * 50.0, p.y * 50.0]);

        // let noise = land;

        let d = 1.0_f64.min(glm::distance(&DVec2::new(0.5, 0.5), &p) * 2.0);
        let grad = glm::smoothstep(0.0, 1.0, 4.0 * (1.0 - d));

        // y = 0.4 - (0.1 - (x - 0.4)) / 0.1

        // let biome = b as f64 / 9.0;

        // (1.0 - voronoi_d.pow(3.0)) * grad
        col * grad
        // noise * grad
    };

    let t0 = Instant::now();

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            buf[i] = (v * 255.0) as u8;
            buf[i + 1] = buf[i];
            buf[i + 2] = buf[i];
        }
    }

    let t1 = Instant::now();

    println!(
        "T: {}",
        (t1 - t0).as_secs_f64() / 1024.0 / 1024.0 * 24.0_f64.powi(3)
    );

    image::save_buffer("noise_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}
