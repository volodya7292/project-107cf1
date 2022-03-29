use common;

fn main() {
    println!("cargo:rerun-if-changed=src/renderer/shaders");
    println!("cargo:rerun-if-changed=res/");

    common::compile_shaders("src/shaders", "res/shaders");
    common::encode_resources("res", "work_dir/resources");
}
