use std::collections::{hash_map, HashMap};
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

    let out_dir = env::var("OUT_DIR").unwrap();
    let ts_path_s = out_dir + "/shader_timestamps";
    let ts_path = Path::new(&ts_path_s);

    // Read timestamp file
    let mut timestamps: HashMap<String, TSMetaData> = match fs::File::open(ts_path) {
        Ok(f) => match serde_yaml::from_reader(f) {
            Ok(x) => x,
            Err(_) => HashMap::new(),
        },
        Err(_) => HashMap::new(),
    };

    let pattern = src_dir.to_str().unwrap().to_owned() + "/**/*";

    for entry in glob::glob(&pattern).unwrap() {
        if let Ok(entry) = entry {
            if entry.is_dir() {
                continue;
            }
            let ext = entry.extension().map_or("", |v| v.to_str().unwrap());

            if ext == "vert" || ext == "frag" || ext == "geom" || ext == "comp" {
                let stripped = entry.strip_prefix(src_dir.to_str().unwrap()).unwrap();
                let dst_path_s =
                    dst_dir.to_str().unwrap().to_owned() + "/" + stripped.to_str().unwrap() + ".spv";
                let dst_path = Path::new(&dst_path_s);

                // Get new metadata
                let entry_metadata = fs::metadata(&entry).unwrap();
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

                let mut cmd = &mut Command::new("glslangValidator");
                cmd = cmd
                    .arg("--spirv-val")
                    .arg("--target-env")
                    .arg("vulkan1.1")
                    .arg("-o")
                    .arg(dst_path_s.clone())
                    .arg("-V")
                    .arg(entry_str);

                if cfg!(debug_assertions) {
                    cmd = cmd.arg("-g").arg("-Od");
                }

                let output = cmd.output().unwrap();
                println!("{}", String::from_utf8_lossy(&output.stdout));

                if !output.status.success() {
                    panic!("Failed to compile shader");
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
                        .arg("-O");

                    let output = cmd.output().unwrap();
                    println!("{}", String::from_utf8_lossy(&output.stdout));

                    if !output.status.success() {
                        panic!("Failed to optimize shader");
                    }
                }
            }
        }
    }

    // Save timestamp file
    fs::write(&ts_path, serde_yaml::to_string(&timestamps).unwrap()).unwrap();
}
