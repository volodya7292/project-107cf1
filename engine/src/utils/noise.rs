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
            max_value += amplitude;

            amplitude *= persistence;
            freq *= 2.0;
        }

        total / max_value
    }
}

pub struct ParamNoise<const D: usize, N: NoiseFn<[f64; D]>> {
    sources: Vec<(N, f64, f64)>,
}

impl<const D: usize, N> ParamNoise<D, N>
where
    N: NoiseFn<[f64; D]> + Seedable + Clone + Debug,
{
    /// `octaves`: array of (frequency, amplitude)
    pub fn new(noise: N, octaves: &[(f64, f64)]) -> Self {
        let sources: Vec<_> = octaves
            .iter()
            .enumerate()
            .map(|(i, params)| {
                (
                    noise.clone().set_seed(noise.seed() + i as u32),
                    params.0,
                    params.1,
                )
            })
            .collect();
        Self { sources }
    }

    pub fn sample(&self, point: SVector<f64, D>) -> f64 {
        let mut total = 0.0;
        let mut max_value = 0.0;

        for (source, octave_freq, octave_amplitude) in &self.sources {
            let value = source.get((point * *octave_freq).as_slice().try_into().unwrap());

            total += value * octave_amplitude;
            max_value += octave_amplitude;
        }

        total / max_value
    }
}
