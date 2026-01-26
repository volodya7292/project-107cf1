use shader_ids::shader_variant;
use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=src/rendering/shaders");
    println!("cargo:rerun-if-changed=res/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.parent().unwrap().parent().unwrap().parent().unwrap();

    common::shader_compiler::compile_shader_bundles(
        Path::new("src/rendering/shaders"),
        Path::new("res/shaders"),
        &out_dir.join("shaders_cache"),
        &shader_variant::ALL.iter().cloned().collect(),
    );

    common::compile_shaders("src/rendering/shaders/post_process", "res/shaders/post_process");

    common::encode_resources("res", &target_dir.join("resources"));
}
