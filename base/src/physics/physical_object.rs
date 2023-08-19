use common::glm::Vec3;

#[derive(Copy, Clone)]
pub struct ObjectMotion {
    mass: f32,
    pub velocity: Vec3,
}

impl ObjectMotion {
    pub fn new(mass: f32, velocity: Vec3) -> Self {
        Self { mass, velocity }
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }

    pub fn update(&mut self, total_force: Vec3, delta_time: f32) {
        // F = m * a
        let accel = total_force / self.mass;

        // acceleration increases velocity over time (v = da/dt)
        self.velocity += accel * delta_time;

        // x = s0 + v0*t + a*t^2/2

        // h = a * t^2 / 2
        // t = sqrt(h/a*2)

        // 100% lethal height = 27m
        // t = 2.16 secs
        // v = 21 m/s
    }
}

// fn calc_velocity(mass: f32, total_force: Vec3, g_accel: f32) -> Vec3 {}
