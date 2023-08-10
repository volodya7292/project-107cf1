use crate::module::ui::color::Color;
use std::ops::{Add, Mul, Sub};

/// Maps linear time to function-specific time. In and out are in range [0; 1].
/// f(0) must return 0 and f(1) must return 1.
pub type TimeFn = fn(t: f32) -> f32;

pub trait Interpolatable: Copy {
    fn interpolate(a: Self, b: Self, f: f32) -> Self;
}

impl<T: Copy + Add<Output = Self> + Sub + Mul<f32, Output = Self>> Interpolatable for T {
    fn interpolate(a: Self, b: Self, f: f32) -> Self {
        a * (1.0 - f) + b * f
    }
}

impl Interpolatable for Color {
    fn interpolate(a: Self, b: Self, f: f32) -> Self {
        Color::from(Interpolatable::interpolate(a.into_raw(), b.into_raw(), f))
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TransitionTarget<T: Interpolatable> {
    value: T,
    time_fn: TimeFn,
    duration: f64,
}

impl<T: Interpolatable> TransitionTarget<T> {
    pub fn new(value: T, duration: f64) -> Self {
        Self {
            value,
            time_fn: FN_EASE_IN_OUT,
            duration,
        }
    }

    pub fn immediate(value: T) -> Self {
        Self {
            value,
            time_fn: FN_EASE_IN_OUT,
            duration: 0.0,
        }
    }

    pub fn with_time_fn(mut self, time_fn: TimeFn) -> Self {
        self.time_fn = time_fn;
        self
    }

    pub fn with_value(mut self, value: T) -> Self {
        self.value = value;
        self
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn duration(&self) -> f64 {
        self.duration
    }
}

impl<T: Interpolatable + Default> Default for TransitionTarget<T> {
    fn default() -> Self {
        Self::immediate(Default::default())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AnimatedValue<T: Interpolatable> {
    start: T,
    target: TransitionTarget<T>,
    curr: T,
    time_passed: f64,
}

impl<T: Interpolatable + Default> Default for AnimatedValue<T> {
    fn default() -> Self {
        Self {
            start: Default::default(),
            target: TransitionTarget {
                value: Default::default(),
                time_fn: FN_EASE_IN_OUT,
                duration: 0.0,
            },
            curr: Default::default(),
            time_passed: 0.0,
        }
    }
}

pub const FN_LINEAR: TimeFn = |t: f32| -> f32 { t };
pub const FN_EASE_IN_OUT: TimeFn =
    |t: f32| -> f32 { 0.5 * (common::utils::smoothstep(t) + common::utils::smootherstep(t)) };
pub const FN_EASE_IN: TimeFn = |t: f32| -> f32 { 2.0 * common::utils::smoothstep(0.5 * t) };
pub const FN_EASE_OUT: TimeFn = |t: f32| -> f32 { 2.0 * common::utils::smoothstep(0.5 + 0.5 * t) - 1.0 };

impl<T: Interpolatable> AnimatedValue<T> {
    /// Returns a finished animation with the target value.
    pub fn immediate(value: T) -> Self {
        Self {
            start: value,
            target: TransitionTarget::immediate(value),
            curr: value,
            time_passed: 0.0,
        }
    }

    /// Smoothly change transition target value.
    pub fn retarget(&mut self, target: TransitionTarget<T>) {
        self.start = self.curr;
        self.target = target;
        self.time_passed = 0.0;
        if target.duration == 0.0 {
            self.curr = target.value;
        }
    }

    /// Advances the transition, applies transformation to `value`.
    /// Returns `true` if the transition is finished, `false` otherwise.
    pub fn advance(&mut self, delta_time: f64) -> bool {
        self.time_passed = (self.time_passed + delta_time).min(self.target.duration);

        let norm_linear_t = if self.target.duration == 0.0 {
            1.0
        } else {
            (self.time_passed / self.target.duration) as f32
        };
        let t = (self.target.time_fn)(norm_linear_t);

        self.curr = T::interpolate(self.start, self.target.value, t);

        self.time_passed == self.target.duration
    }

    pub fn is_finished(&self) -> bool {
        self.time_passed == self.target.duration
    }

    pub fn time_passed(&self) -> f64 {
        self.time_passed
    }

    pub fn current(&self) -> &T {
        &self.curr
    }

    pub fn target(&self) -> &TransitionTarget<T> {
        &self.target
    }
}

impl<T: Interpolatable> From<T> for TransitionTarget<T> {
    fn from(value: T) -> Self {
        Self::immediate(value)
    }
}

impl<T: Interpolatable> From<T> for AnimatedValue<T> {
    fn from(value: T) -> Self {
        Self::immediate(value)
    }
}
