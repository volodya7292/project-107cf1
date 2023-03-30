use common::glm::{Vec4};
use std::ops::{Add, Mul, Sub};

/// Maps linear time to function-specific time. In and out are in range [0; 1].
/// f(0) must return 0 and f(1) must return 1.
pub type TimeFn = fn(t: f32) -> f32;

pub trait TransValue: Copy + Add<Output = Self> + Sub + Mul<f32, Output = Self> {}

impl TransValue for Vec4 {}

#[derive(Clone)]
pub struct Transition<T: TransValue> {
    start: T,
    target: T,
    time_fn: TimeFn,
    duration: f64,
    time_passed: f64,
}

pub const FN_LINEAR: TimeFn = |t: f32| -> f32 { t };
pub const FN_EASE_IN_OUT: TimeFn =
    |t: f32| -> f32 { 0.5 * (common::utils::smoothstep(t) + common::utils::smootherstep(t)) };
pub const FN_EASE_IN: TimeFn = |t: f32| -> f32 { 2.0 * common::utils::smoothstep(0.5 * t) };
pub const FN_EASE_OUT: TimeFn = |t: f32| -> f32 { 2.0 * common::utils::smoothstep(0.5 + 0.5 * t) - 1.0 };

impl<T: TransValue> Transition<T> {
    pub fn new(start: T, target: T, duration: f64, time_fn: TimeFn) -> Self {
        Self {
            start,
            target,
            time_fn,
            duration,
            time_passed: 0.0,
        }
    }

    /// Returns a finished animation with the target value.
    pub fn none(value: T) -> Self {
        Self {
            start: value,
            target: value,
            time_fn: FN_EASE_IN_OUT,
            duration: 1.0,
            time_passed: 1.0,
        }
    }

    /// Advances the transition, applies transformation to `value`.
    /// Returns `true` if the transition is finished, `false` otherwise.
    pub fn advance(&mut self, value: &mut T, delta_time: f64) -> bool {
        self.time_passed = (self.time_passed + delta_time).min(self.duration);

        let norm_linear_t = (self.time_passed / self.duration) as f32;
        let t = (self.time_fn)(norm_linear_t);

        *value = self.start * (1.0 - t) + self.target * t;

        self.time_passed == self.duration
    }
}
