use nalgebra::SVector;
use noise::{NoiseFn, SuperSimplex};
use std::convert::TryInto;

pub trait ParamNoise<const D: usize> {
    fn sample(&self, point: SVector<f64, D>, freq: f64, octaves: f64, persistence: f64) -> f64;
}

impl<const D: usize> ParamNoise<D> for SuperSimplex
where
    SuperSimplex: NoiseFn<[f64; D]>,
{
    fn sample(&self, point: SVector<f64, D>, freq: f64, octaves: f64, persistence: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = freq;
        let mut amp = 1.0;
        let mut max_value = 0.0;

        for i in 0..(octaves.ceil() as u32) {
            let m = amp * (octaves - i as f64).min(1.0);
            let v = (self.get((point * freq).as_slice().try_into().unwrap()) + 1.0) * 0.5;
            total += v * m;
            max_value += m;
            amp *= persistence;
            freq *= 2.0;
        }

        total / max_value
    }
}
