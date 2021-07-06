use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::marker::PhantomData;

pub struct ValueNoise<T> {
    main_state: u64,
    _ty: PhantomData<T>,
}

pub struct State(pub u64);

impl State {
    pub fn next(mut self, perm: u64) -> State {
        self.0 ^= Xoshiro256PlusPlus::seed_from_u64(perm)
            .gen::<u64>()
            .wrapping_add(0x9e3779b9)
            .wrapping_add(self.0 << 6)
            .wrapping_add(self.0 >> 2);
        self
    }

    pub fn rng(&self) -> impl rand::Rng {
        Xoshiro256PlusPlus::seed_from_u64(self.0)
    }
}

impl<T> ValueNoise<T> {
    pub fn new(seed: u64) -> ValueNoise<T> {
        ValueNoise {
            main_state: Xoshiro256PlusPlus::seed_from_u64(seed).gen::<u64>(),
            _ty: Default::default(),
        }
    }

    pub fn state(&self) -> State {
        State(self.main_state)
    }
}
