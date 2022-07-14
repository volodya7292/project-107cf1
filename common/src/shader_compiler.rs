use std::collections::{hash_map, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;
use std::{env, fs};

#[derive(serde::Serialize, serde::Deserialize)]
struct TSMetaData {
    pub modified: SystemTime,
}

pub fn compile_shaders<P: AsRef<Path>>(src_dir: P, dst_dir: P) {
    let src_dir = src_dir.as_ref();
    let dst_dir = dst_dir.as_ref();

    if !dst_dir.exists() {
        fs::create_dir(dst_dir).unwrap();
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let ts_path_s = out_dir + "/shader_timestamps";
    let ts_path = Path::new(&ts_path_s);

    let mut log_file = File::create(dst_dir.join("output.log")).unwrap();

    // Read timestamp file
    let mut timestamps: HashMap<String, TSMetaData> = match fs::File::open(ts_path) {
        Ok(f) => match serde_yaml::from_reader(f) {
            Ok(x) => x,
            Err(_) => HashMap::new(),
        },
        Err(_) => HashMap::new(),
    };

    for entry in src_dir.read_dir().unwrap() {
        if let Ok(entry) = entry {
            let entry = entry.path();
            if entry.is_dir() {
                continue;
            }

            let ext = entry.extension().map_or("", |v| v.to_str().unwrap());

            if ext != "vert" && ext != "frag" && ext != "comp" && ext != "hlsl" {
                continue;
            }

            let stripped = entry.strip_prefix(src_dir.to_str().unwrap()).unwrap();
            let dst_path_s = dst_dir.to_str().unwrap().to_owned() + "/" + stripped.to_str().unwrap() + ".spv";
            let dst_path = Path::new(&dst_path_s);

            // Get new metadata
            let entry_metadata = entry.metadata().unwrap();
            let new_ts = TSMetaData {
                modified: entry_metadata.modified().unwrap(),
            };

            // Check for modification
            let modified;
            match timestamps.entry(entry.to_str().unwrap().to_owned()) {
                hash_map::Entry::Occupied(a) => {
                    modified = &new_ts.modified > &a.get().modified;
                }
                hash_map::Entry::Vacant(a) => {
                    a.insert(new_ts);
                    modified = true;
                }
            }

            if !modified && dst_path.exists() {
                continue;
            }

            fs::create_dir_all(dst_path.parent().unwrap()).unwrap();

            let entry_str = entry.to_str().unwrap();

            let mut cmd = if ext == "hlsl" {
                let name = entry.file_name().unwrap().to_str().unwrap();
                let name_without_hlsl = Path::new(name.strip_suffix(".hlsl").unwrap());
                let ty = if let Some(ty) = name_without_hlsl.extension() {
                    ty.to_str().unwrap()
                } else {
                    continue;
                };
                let target = match ty {
                    "vert" => "vs_6_0",
                    "frag" => "ps_6_0",
                    "comp" => "cs_6_0",
                    _ => panic!("Unsupported shader type"),
                };
                let mut cmd = Command::new("dxc");

                cmd.arg("-spirv")
                    .arg("-fvk-use-scalar-layout")
                    .arg("-fspv-target-env=vulkan1.1")
                    .arg("-HV")
                    .arg("2021")
                    .arg("-T")
                    .arg(target)
                    .arg("-Fo")
                    .arg(dst_path_s.clone())
                    .arg(entry_str);
                cmd
            } else {
                let mut cmd = Command::new("glslangValidator");
                cmd.arg("--spirv-val")
                    .arg("--target-env")
                    .arg("vulkan1.1")
                    .arg("-e")
                    .arg("main")
                    .arg("-o")
                    .arg(dst_path_s.clone())
                    .arg("-V")
                    .arg(entry_str);

                if cfg!(debug_assertions) {
                    cmd.arg("-g").arg("-Od");
                }
                cmd
            };

            let output = cmd.output().unwrap();
            log_file.write(&output.stdout).unwrap();
            log_file.write(&output.stderr).unwrap();

            print!("{}", String::from_utf8_lossy(&output.stdout));
            print!("{}", String::from_utf8_lossy(&output.stderr));

            if !output.status.success() {
                panic!("Failed to compile shader: {:?}", cmd);
            }

            // Optimize shader
            if !cfg!(debug_assertions) {
                let mut cmd = &mut Command::new("spirv-opt");
                cmd = cmd
                    .arg(dst_path_s.clone())
                    .arg("-o")
                    .arg(dst_path_s.clone())
                    .arg("--target-env=vulkan1.1")
                    .arg("--skip-validation")
                    .arg("--preserve-bindings")
                    .arg("--scalar-block-layout")
                    .arg("-O");

                let output = cmd.output().unwrap();
                log_file.write(&output.stdout).unwrap();
                print!("{}", String::from_utf8_lossy(&output.stdout));

                if !output.status.success() {
                    panic!("Failed to optimize shader");
                }
            }
        }
    }

    // Save timestamp file
    fs::write(&ts_path, serde_yaml::to_string(&timestamps).unwrap()).unwrap();
}
