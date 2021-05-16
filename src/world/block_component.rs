use std::mem;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Facing {
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl Facing {
    pub fn from_u8(v: u8) -> Facing {
        assert!(v < 6);
        unsafe { mem::transmute(v) }
    }
}
