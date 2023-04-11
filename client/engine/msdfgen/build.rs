use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=cxx");
    println!("cargo:rustc-link-lib=msdfgen");

    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap();

    let files: Vec<_> = Path::new("cxx/core")
        .read_dir()
        .unwrap()
        .map(|v| v.unwrap().path())
        .filter(|v| v.extension().unwrap() == "cpp")
        .collect();

    let mut build = cxx_build::bridge("src/lib.rs");
    build.files(files).include("cxx").flag("-w").warnings(false);

    if target_env != "msvc" {
        build.flag("-std=c++17");
    }

    build.compile("msdfgen");
}
