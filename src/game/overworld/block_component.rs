use nalgebra_glm::{I32Vec3, U32Vec3};
use std::mem;

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Facing {
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl Facing {
    pub const DIRECTIONS: [I32Vec3; 6] = [
        I32Vec3::new(-1, 0, 0),
        I32Vec3::new(1, 0, 0),
        I32Vec3::new(0, -1, 0),
        I32Vec3::new(0, 1, 0),
        I32Vec3::new(0, 0, -1),
        I32Vec3::new(0, 0, 1),
    ];

    pub const fn direction(&self) -> I32Vec3 {
        Self::DIRECTIONS[*self as usize]
    }

    pub fn from_u8(v: u8) -> Facing {
        assert!(v < 6);
        unsafe { mem::transmute(v) }
    }

    pub fn from_direction(dir: I32Vec3) -> Option<Facing> {
        if dir.sum().abs() != 1 {
            None
        } else {
            Some(Self::from_u8(
                (dir.x == 1) as u8
                    + dir.y.abs() as u8 * 2
                    + (dir.y == 1) as u8
                    + dir.z.abs() as u8 * 4
                    + (dir.z == 1) as u8,
            ))
        }
    }

    pub fn mirror(&self) -> Facing {
        let id = *self as u8;
        Facing::from_u8((id & !1) + (id + 1) % 2)
    }
}
