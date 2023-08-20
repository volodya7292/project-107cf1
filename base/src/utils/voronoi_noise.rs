use crate::utils::white_noise::WhiteNoise;
use common::glm;
use glm::{DVec2, I64Vec2};
use rand::Rng;
use std::ops::Range;

const MIN_DIST_BETWEEN_POINTS: f64 = 0.2;
// Note: clamp to 0.9 to avoid generating thin geometries
const POINT_RANGE: Range<f64> = (-1.0 + MIN_DIST_BETWEEN_POINTS / 2.0)..(1.0 - MIN_DIST_BETWEEN_POINTS / 2.0);

pub struct VoronoiNoise2D {
    white_noise: WhiteNoise,
}

impl VoronoiNoise2D {
    pub fn new() -> Self {
        Self {
            white_noise: WhiteNoise::new(0),
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.white_noise = WhiteNoise::new(seed);
        self
    }

    /// Returns closest point to `p` and distance to it.
    pub fn sample(&self, p: DVec2) -> (DVec2, f64) {
        let r: DVec2 = p.map(|v| v.round());
        let ri: I64Vec2 = glm::convert_unchecked(r);

        let mut min_p = DVec2::default();
        let mut min_dist = f64::MAX;

        for x in (ri.x - 1)..=(ri.x + 1) {
            for y in (ri.y - 1)..=(ri.y + 1) {
                let mut rng = self.white_noise.state().next(x).next(y).rng();

                let jitter = DVec2::new(rng.gen_range(POINT_RANGE), rng.gen_range(POINT_RANGE));
                let v = glm::convert::<_, DVec2>(glm::vec2(x, y)) + jitter * 0.39614353;

                let dist = glm::distance2(&v, &p);

                if dist < min_dist {
                    min_p = v;
                    min_dist = dist;
                }
            }
        }

        (min_p, min_dist.sqrt())
    }

    // pub fn sample_3x3(&mut self, p: DVec3) -> [[[DVec3; 3]; 3]; 3] {
    //     let r: DVec3 = p.map(|v| v.floor());
    //     let ri: I64Vec3 = glm::convert_unchecked(r);
    //
    //     let mut result: [[[DVec3; 3]; 3]; 3] = Default::default();
    //
    //     for x in 0..3 {
    //         for y in 0..3 {
    //             for z in 0..3 {
    //                 let g = ri + glm::convert::<_, I64Vec3>(glm::vec3(x, y, z)).add_scalar(-1);
    //                 let mut rng = self.white_noise.state().next(g.x).next(g.y).next(g.z).rng();
    //
    //                 let jitter = DVec3::new(
    //                     rng.gen_range(POINT_RANGE),
    //                     rng.gen_range(POINT_RANGE),
    //                     rng.gen_range(POINT_RANGE),
    //                 );
    //                 let v = glm::convert::<_, DVec3>(g) + jitter * 0.39614353;
    //
    //                 result[x][y][z] = v;
    //             }
    //         }
    //     }
    //
    //     result
    // }
}

impl Default for VoronoiNoise2D {
    fn default() -> Self {
        Self::new()
    }
}
