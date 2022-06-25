use std::path::Path;

fn main() {
    println!("cargo:rustc-link-lib=msdfgen");

    let files: Vec<_> = Path::new("cxx/core")
        .read_dir()
        .unwrap()
        .map(|v| v.unwrap().path())
        .filter(|v| v.extension().unwrap() == "cpp")
        .collect();

    cxx_build::bridge("src/lib.rs")
        .files(files)
        .include("cxx")
        .flag("-w")
        .flag("-std=c++17")
        .compile("msdfgen");
}
