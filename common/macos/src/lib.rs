#![cfg(target_os = "macos")]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unnecessary_transmutes)]
#![allow(unused_unsafe)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
