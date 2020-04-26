use std::ffi::{CStr, CString};
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
