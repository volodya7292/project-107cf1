use nalgebra::SVector;
use noise::{NoiseFn, Seedable};
use std::convert::TryInto;
use std::fmt::Debug;

pub struct HybridNoise<const D: usize, const OCTAVES: usize, N: NoiseFn<[f64; D]>> {
    sources: [N; OCTAVES],
}

impl<const D: usize, const OCTAVES: usize, N> HybridNoise<D, OCTAVES, N>
where
    N: NoiseFn<[f64; D]> + Seedable + Clone + Debug,
{
    pub fn new(noise: N) -> Self {
        let sources: Vec<_> = (0..OCTAVES)
            .map(|i| noise.clone().set_seed(noise.seed() + i as u32))
            .collect();
        Self {
            sources: sources.try_into().unwrap(),
        }
    }

    pub fn sample(&self, point: SVector<f64, D>, freq: f64, persistence: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = freq;
        let mut amplitude = 1.0;
        let mut max_value = 0.0;

        for source in &self.sources {
            let value = source.get((point * freq).as_slice().try_into().unwrap());

            total += value * amplitude;
            amplitude *= persistence;
            freq *= 2.0;

            max_value += amplitude;
        }

        total / max_value
    }
}
