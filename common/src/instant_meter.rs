use std::fmt::{Display, Formatter};
use std::time::Instant;

struct TimePoint {
    name: String,
    time: Instant,
}

pub struct InstantMeter {
    steps: Vec<TimePoint>,
}

impl InstantMeter {
    pub fn new() -> Self {
        Self {
            steps: Vec::with_capacity(128),
        }
    }

    pub fn next(&mut self, step: impl Into<String>) {
        self.steps.push(TimePoint {
            name: step.into(),
            time: Instant::now(),
        });
    }

    pub fn finish(&mut self) {
        self.next("");
    }

    pub fn print(&self) {
        println!("{}", self);
    }
}

impl Display for InstantMeter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("InstantMeter [")?;

        let filtered: Vec<_> = self.steps.windows(2).filter(|p| !p[0].name.is_empty()).collect();

        for (i, points) in filtered.iter().enumerate() {
            let p0 = &points[0];
            let p1 = &points[1];

            let delta = p1.time - p0.time;
            let secs = delta.as_secs_f64();

            f.write_fmt(format_args!("{} = {:.9}", &p0.name, secs))?;

            if i < filtered.len() - 1 {
                f.write_str(", ")?;
            }
        }

        f.write_str("]")?;

        Ok(())
    }
}
