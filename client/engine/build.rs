fn main() {
    println!("cargo:rerun-if-changed=shaders/");

    common::compile_shaders("shaders", "shaders/build");
}
