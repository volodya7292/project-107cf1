use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub trait AsSeed {
    fn as_u64_seed(&self) -> u64;
}

impl AsSeed for u64 {
    fn as_u64_seed(&self) -> u64 {
        *self
    }
}

impl AsSeed for i64 {
    fn as_u64_seed(&self) -> u64 {
        u64::from_ne_bytes(self.to_ne_bytes())
    }
}

impl AsSeed for f64 {
    fn as_u64_seed(&self) -> u64 {
        u64::from_ne_bytes(self.to_ne_bytes())
    }
}

pub struct WhiteNoise {
    main_state: u64,
}

pub struct State(pub u64);

impl State {
    pub fn next<S: AsSeed>(mut self, perm: S) -> State {
        self.0 ^= Xoshiro256PlusPlus::seed_from_u64(perm.as_u64_seed())
            .gen::<u64>()
            .wrapping_add(0x9e3779b9)
            .wrapping_add(self.0 << 6)
            .wrapping_add(self.0 >> 2);
        self
    }

    pub fn rng(&self) -> impl Rng {
        Xoshiro256PlusPlus::seed_from_u64(self.0)
    }
}

impl WhiteNoise {
    pub fn new(seed: u64) -> WhiteNoise {
        WhiteNoise {
            main_state: Xoshiro256PlusPlus::seed_from_u64(seed).gen::<u64>(),
        }
    }

    pub fn state(&self) -> State {
        State(self.main_state)
    }
}
