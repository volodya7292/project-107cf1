use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::collections::{hash_map, BTreeSet, HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::SystemTime;
use std::{env, fs};

#[derive(Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct ShaderVariantConfig {
    definitions: BTreeSet<String>,
}

impl ShaderVariantConfig {
    pub fn new(definitions: Vec<String>) -> Self {
        for def in &definitions {
            assert!(def.chars().count() > 0);
            assert!(def
                .chars()
                .all(|v| (v.is_ascii_alphanumeric() || v == '_') && !v.is_whitespace()));
            assert_eq!(def, &def.to_uppercase());
        }
        Self {
            definitions: definitions.into_iter().collect(),
        }
    }

    pub fn uid(&self) -> String {
        let defs = self.definitions.iter().cloned().collect::<Vec<_>>();
        defs.join("-")
    }
}

/// Returns `true` if compilation is successful.
pub fn compile_shader(
    src_path: &Path,
    dst_path: &Path,
    config: &ShaderVariantConfig,
    log: &mut dyn Write,
) -> Result<(), String> {
    let entry_str = src_path.to_str().unwrap();

    let mut cmd = Command::new("glslangValidator");
    cmd.arg("--spirv-val")
        .arg("--target-env")
        .arg("vulkan1.2")
        .arg("-e")
        .arg("main")
        .arg("-o")
        .arg(dst_path)
        .arg("-V")
        .arg(entry_str);

    for def in &config.definitions {
        cmd.arg(format!("-D{def}=1"));
    }
    if cfg!(debug_assertions) {
        cmd.arg("-g").arg("-Od");
    }

    let output = cmd.output().unwrap();
    log.write_all(&output.stdout).unwrap();
    log.write_all(&output.stderr).unwrap();

    print!("{}", String::from_utf8_lossy(&output.stdout));
    print!("{}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        return Err(format!("Failed to compile shader: {:?}", cmd));
    }

    // Optimize shader
    let mut cmd = &mut Command::new("spirv-opt");
    cmd = cmd
        .arg(dst_path.clone())
        .arg("-o")
        .arg(dst_path.clone())
        .arg("--target-env=vulkan1.2")
        .arg("--eliminate-dead-variables")
        .arg("--eliminate-dead-input-components")
        .arg("--skip-validation")
        .arg("--preserve-bindings")
        .arg("--scalar-block-layout")
        .arg("-O");

    let output = cmd.output().unwrap();
    log.write_all(&output.stdout).unwrap();
    print!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.status.success() {
        return Err("Failed to optimize shader".to_string());
    }

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct TSMetaData {
    pub modified: SystemTime,
}

pub fn compile_shader_cached(
    src_path: &Path,
    dst_path: &Path,
    config: &ShaderVariantConfig,
    timestamps: &mut HashMap<String, TSMetaData>,
    force_recompile: bool,
    log: &mut dyn Write,
) -> Result<(), String> {
    // Get new metadata
    let entry_metadata = src_path.metadata().unwrap();
    let new_ts = TSMetaData {
        modified: entry_metadata.modified().unwrap(),
    };

    // Check for modification
    let modified = match timestamps.entry(src_path.to_str().unwrap().to_owned()) {
        hash_map::Entry::Occupied(a) => new_ts.modified > a.get().modified,
        hash_map::Entry::Vacant(a) => {
            a.insert(new_ts);
            true
        }
    };

    if !force_recompile && !modified && dst_path.exists() {
        return Ok(());
    }

    compile_shader(src_path, dst_path, config, log)
}

pub fn compile_shader_cached_to_memory(
    src_path: &Path,
    cache_dir: &Path,
    config: &ShaderVariantConfig,
    timestamps: &mut HashMap<String, TSMetaData>,
    force_recompile: bool,
    log: &mut dyn Write,
) -> Result<Vec<u8>, String> {
    let src_dir = src_path.parent().unwrap();
    let stripped_filename = src_path.strip_prefix(src_dir).unwrap();
    let dst_path = cache_dir.join(format!(
        "{}__{}.spv",
        stripped_filename.to_str().unwrap(),
        config.uid(),
    ));

    compile_shader_cached(src_path, &dst_path, config, timestamps, force_recompile, log)?;

    let code = fs::read(&dst_path).unwrap();
    Ok(code)
}

pub type ShaderSpirvCode = ByteBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct ShaderBundle {
    pub variants: HashMap<ShaderVariantConfig, ShaderSpirvCode>,
}

pub fn compile_shader_bundle(
    src_path: &Path,
    cache_dir: &Path,
    variant_configs: &HashSet<ShaderVariantConfig>,
    timestamps: &mut HashMap<String, TSMetaData>,
    force_recompile: bool,
    log: &mut dyn Write,
) -> ShaderBundle {
    let variants: HashMap<ShaderVariantConfig, ShaderSpirvCode> = variant_configs
        .iter()
        .map(|config| {
            let code = compile_shader_cached_to_memory(
                src_path,
                cache_dir,
                config,
                timestamps,
                force_recompile,
                log,
            )
            .unwrap();
            (config.clone(), ByteBuf::from(code))
        })
        .collect();

    ShaderBundle { variants }
}

pub fn save_shader_bundle<P: AsRef<Path>>(path: P, bundle: &ShaderBundle) {
    let data = bincode::serialize(&bundle).unwrap();
    fs::write(path, data).unwrap();
}

pub fn read_shader_bundle(data: &[u8]) -> ShaderBundle {
    bincode::deserialize::<ShaderBundle>(data).unwrap()
}

pub fn is_shader_source_ext(ext: &str) -> bool {
    matches!(ext, "vert" | "frag" | "comp" | "hlsl")
}

pub fn compile_shader_bundles(
    src_dir: &Path,
    dst_dir: &Path,
    cache_dir: &Path,
    variant_configs: &HashSet<ShaderVariantConfig>,
) {
    if !dst_dir.exists() {
        fs::create_dir(dst_dir).unwrap();
    }
    if !cache_dir.exists() {
        fs::create_dir(cache_dir).unwrap();
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let ts_path_s = out_dir + "/shader_timestamps";
    let ts_path = Path::new(&ts_path_s);

    let mut log_file = File::create(dst_dir.join("output.log")).unwrap();

    let mut timestamps = File::open(ts_path)
        .map(|f| serde_yaml::from_reader::<_, HashMap<String, TSMetaData>>(f).ok())
        .ok()
        .flatten()
        .unwrap_or(Default::default());

    // Compile new bundles
    for entry in src_dir.read_dir().unwrap().filter_map(Result::ok) {
        let entry = entry.path();
        if entry.is_dir() {
            continue;
        }

        let ext = entry.extension().map_or("", |v| v.to_str().unwrap());
        if !is_shader_source_ext(ext) {
            continue;
        }

        let bundle_path = {
            let src_dir = entry.parent().unwrap();
            let stripped_filename = entry.strip_prefix(src_dir).unwrap();
            dst_dir.join(format!("{}.b", stripped_filename.to_str().unwrap()))
        };

        let force_recompile = !bundle_path.exists();
        let bundle = compile_shader_bundle(
            &entry,
            cache_dir,
            variant_configs,
            &mut timestamps,
            force_recompile,
            &mut log_file,
        );
        save_shader_bundle(bundle_path, &bundle);
    }

    // Cleanup old bundles
    for entry in dst_dir.read_dir().unwrap().filter_map(Result::ok) {
        let entry = entry.path();
        if entry.is_dir() {
            continue;
        }

        if entry.extension() != Some(OsStr::new("b")) {
            continue;
        }

        let src_path = {
            let entry_without_ext = entry.with_extension("");
            let stripped_filename = entry_without_ext.strip_prefix(dst_dir).unwrap();
            src_dir.join(stripped_filename)
        };

        if !src_path.exists() {
            fs::remove_file(entry).unwrap();
        }
    }

    // Save timestamps file
    fs::write(ts_path, serde_yaml::to_string(&timestamps).unwrap()).unwrap();
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
    let mut timestamps = File::open(ts_path)
        .map(|f| serde_yaml::from_reader::<_, HashMap<String, TSMetaData>>(f).ok())
        .ok()
        .flatten()
        .unwrap_or(Default::default());

    for entry in src_dir.read_dir().unwrap() {
        let Ok(entry) = entry else {
            continue;
        };
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
        let modified = match timestamps.entry(entry.to_str().unwrap().to_owned()) {
            hash_map::Entry::Occupied(a) => new_ts.modified > a.get().modified,
            hash_map::Entry::Vacant(a) => {
                a.insert(new_ts);
                true
            }
        };

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
                .arg("-fspv-target-env=vulkan1.2")
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
                .arg("vulkan1.2")
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
        log_file.write_all(&output.stdout).unwrap();
        log_file.write_all(&output.stderr).unwrap();

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
                .arg("--target-env=vulkan1.2")
                .arg("--skip-validation")
                .arg("--preserve-bindings")
                .arg("--scalar-block-layout")
                .arg("-O");

            let output = cmd.output().unwrap();
            log_file.write_all(&output.stdout).unwrap();
            print!("{}", String::from_utf8_lossy(&output.stdout));

            if !output.status.success() {
                panic!("Failed to optimize shader");
            }
        }
    }

    // Save timestamp file
    fs::write(ts_path, serde_yaml::to_string(&timestamps).unwrap()).unwrap();
}
