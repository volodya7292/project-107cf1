use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let target = env::var("CARGO_CFG_TARGET_ENV").unwrap();
    let vulkan_sdk_path = env::var("VULKAN_SDK").unwrap();
    let vulkan_include_path = format!("{}/include", vulkan_sdk_path);

    let mut build = cc::Build::new();
    build.include(&vulkan_include_path).file("wrapper.cpp").cpp(true).flag("-w").warnings(false);

    if target != "msvc" {
        build.flag("-std=c++17");
    }

    // Compile and link the library
    build.compile("vma");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-v")
        .clang_arg(format!("-I{}", &vulkan_include_path))
        .allowlist_function("vma.*")
        .allowlist_function("PFN_vma.*")
        .allowlist_type("Vma.*")
        .blocklist_type("__darwin_.*")
        .size_t_is_usize(true)
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
