use shader_ids::shader_variant;
use std::{env, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=src/rendering/shaders");
    println!("cargo:rerun-if-changed=res/");

    common::shader_compiler::compile_shader_bundles(
        Path::new("src/rendering/shaders"),
        Path::new("res/shaders"),
        Path::new(&format!("{}/shaders_cache", &env::var("OUT_DIR").unwrap())),
        &shader_variant::ALL.iter().cloned().collect(),
    );

    common::compile_shaders("src/rendering/shaders/post_process", "res/shaders/post_process");

    common::encode_resources("res", "work_dir/resources");
}
