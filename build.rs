fn main() {
    println!("cargo:rerun-if-changed=src/rendering/shaders");
    println!("cargo:rerun-if-changed=res/");

    common::compile_shaders("src/rendering/shaders", "res/shaders");
    common::encode_resources("res", "work_dir/resources");
}
