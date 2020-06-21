use std::ffi::{CStr, CString};
use std::ops;
use std::os::raw::c_char;

pub(crate) unsafe fn c_ptr_to_string(ptr: *const c_char) -> String {
    String::from(CStr::from_ptr(ptr).to_str().unwrap())
}

pub(crate) fn filter_names(v: &Vec<String>, f: &[&str], required: bool) -> Result<Vec<CString>, String> {
    let mut err_str = "".to_string();

    let v = f
        .iter()
        .filter_map(|&name| {
            if v.contains(&name.to_string()) {
                Some(CString::new(name).unwrap())
            } else {
                if required {
                    err_str.push_str(format!("{} not available!\n", name).as_ref());
                }
                None
            }
        })
        .collect();

    if err_str.is_empty() {
        Ok(v)
    } else {
        Err(err_str)
    }
}

/// Make number multiple of another number
/// # Examples
/// make_mul_of(7, 4) -> 8;
/// make_mul_of(8, 4) -> 8;
/// make_mul_of(67, 8) -> 72
pub(crate) fn make_mul_of_u64(number: u64, multiplier: u64) -> u64 {
    ((number + multiplier - 1) / multiplier) * multiplier
}

macro_rules! vk_bitflags_impl {
    ($name: ident, $flag_type: ty) => {
        impl Default for $name {
            fn default() -> $name {
                $name(<$flag_type>::default())
            }
        }
        impl $name {
            #[inline]
            pub const fn empty() -> $name {
                $name(<$flag_type>::empty())
            }
            #[inline]
            pub const fn all() -> $name {
                $name(<$flag_type>::all())
            }
            #[inline]
            pub const fn from_raw(x: $flag_type) -> Self {
                $name(x)
            }
            #[inline]
            pub const fn as_raw(self) -> $flag_type {
                self.0
            }
            #[inline]
            pub fn is_empty(self) -> bool {
                self == $name::empty()
            }
            #[inline]
            pub fn is_all(self) -> bool {
                self & $name::all() == $name::all()
            }
            #[inline]
            pub fn intersects(self, other: $name) -> bool {
                self & other != $name::empty()
            }
            #[inline]
            pub fn contains(self, other: $name) -> bool {
                self & other == other
            }
        }
        impl ::std::ops::BitOr for $name {
            type Output = $name;
            #[inline]
            fn bitor(self, rhs: $name) -> $name {
                $name(self.0 | rhs.0)
            }
        }
        impl ::std::ops::BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: $name) {
                *self = *self | rhs
            }
        }
        impl ::std::ops::BitAnd for $name {
            type Output = $name;
            #[inline]
            fn bitand(self, rhs: $name) -> $name {
                $name(self.0 & rhs.0)
            }
        }
        impl ::std::ops::BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: $name) {
                *self = *self & rhs
            }
        }
        impl ::std::ops::BitXor for $name {
            type Output = $name;
            #[inline]
            fn bitxor(self, rhs: $name) -> $name {
                $name(self.0 ^ rhs.0)
            }
        }
        impl ::std::ops::BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: $name) {
                *self = *self ^ rhs
            }
        }
        impl ::std::ops::Sub for $name {
            type Output = $name;
            #[inline]
            fn sub(self, rhs: $name) -> $name {
                self & !rhs
            }
        }
        impl ::std::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: $name) {
                *self = *self - rhs
            }
        }
        impl ::std::ops::Not for $name {
            type Output = $name;
            #[inline]
            fn not(self) -> $name {
                self ^ $name::all()
            }
        }
    };
}
