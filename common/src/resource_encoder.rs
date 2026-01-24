use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Formatter;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::string::String;
use std::time::SystemTime;
use std::{env, fmt, fs};

/**
File structure
    | <u64> header size | resource entry hierarchy | resources data |

Resource entry structure
    | <UTF-8 String> name | <char> null | <u64> offset | <u64> size |
 */

struct ResFileStructure {
    entries: Vec<EntryInfo>,
    header_size: u32,
    data_size: u64,
    modified: bool,
}

impl fmt::Debug for ResFileStructure {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResFileStructure")
            .field("entries", &self.entries)
            .field("header_size", &self.header_size)
            .field("data_size", &self.data_size)
            .finish()
    }
}

#[derive(Clone)]
struct EntryInfo {
    path: PathBuf,
    modified: bool,
    name: String,
    // dir -> 0, file -> 1
    offset: u64,
    // dir -> entry count, file -> file size
    size: u64,
}

impl fmt::Debug for EntryInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntryInfo")
            .field("path", &self.path)
            .field("name", &self.name)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

fn read_resources(read_dir: fs::ReadDir, timestamps: &mut HashMap<String, SystemTime>) -> ResFileStructure {
    let mut res_file = ResFileStructure {
        entries: vec![],
        header_size: 0,
        data_size: 0,
        modified: false,
    };

    for dir_entry_r in read_dir {
        let dir_entry = dir_entry_r.unwrap();
        let dir_path = dir_entry.path();

        let entry_info;

        if dir_path.is_dir() {
            entry_info = EntryInfo {
                path: dir_path.clone(),
                modified: false,
                name: dir_entry.file_name().into_string().unwrap(),
                offset: 0,
                size: dir_path.read_dir().unwrap().count() as u64,
            };
            res_file.entries.push(entry_info.clone());

            let mut res_file_temp = read_resources(dir_path.read_dir().unwrap(), timestamps);
            res_file.header_size += res_file_temp.header_size;
            res_file.data_size += res_file_temp.data_size;
            res_file.entries.append(&mut res_file_temp.entries);
            if res_file_temp.modified {
                res_file.modified = true;
            }
        } else {
            let metadata = dir_entry.metadata().unwrap();
            let file_size = metadata.len();
            let modified_ts = metadata.modified().unwrap();

            let path_canon = dir_path.canonicalize().unwrap();

            let modified;
            match timestamps.entry(path_canon.to_str().unwrap().to_owned()) {
                Entry::Occupied(o) => {
                    if &modified_ts > o.get() {
                        modified = true;
                        res_file.modified = true;
                    } else {
                        modified = false;
                    }
                }
                Entry::Vacant(v) => {
                    v.insert(modified_ts);
                    modified = true;
                    res_file.modified = true;
                }
            }

            entry_info = EntryInfo {
                path: dir_path.clone(),
                modified,
                name: dir_entry.file_name().into_string().unwrap(),
                offset: 1,
                size: file_size,
            };

            res_file.entries.push(entry_info.clone());

            res_file.data_size += file_size;
        }

        res_file.header_size += 8 * 2 + (entry_info.name.len() as u32 + 1);
    }

    res_file
}

pub fn encode_resources<P: AsRef<Path>>(resources_path: P, output_file_path: P) {
    let resources_path = resources_path.as_ref();
    let output_file_path = output_file_path.as_ref();

    let out_dir_path = output_file_path.parent().unwrap();
    let mut ts_path = env::temp_dir();
    ts_path.push("utx_res_encoder_ts");

    if !resources_path.exists() {
        panic!(
            "Resources directory does not exist: {}",
            resources_path.to_str().unwrap()
        );
    }

    // Read timestamp file
    let mut timestamps: HashMap<String, SystemTime> = fs::read(&ts_path)
        .map(|v| {
            postcard::from_bytes::<HashMap<String, SystemTime>>(&v).unwrap_or_else(|err| {
                println!("Cannot read timestamp file ({err}), regenerating...");
                Default::default()
            })
        })
        .unwrap_or(Default::default());

    // Create destination directories
    if !out_dir_path.exists() {
        fs::create_dir_all(out_dir_path).unwrap_or_else(|_| {
            panic!(
                "Cannot create output directory: {}",
                output_file_path.to_str().unwrap()
            )
        });
    }

    // Read resources
    let mut res_file_struct = read_resources(resources_path.read_dir().unwrap(), &mut timestamps);
    println!("{:#?}", &res_file_struct);

    res_file_struct.header_size += 4; // + sizeof(u32) for header size var
    let res_file_size = res_file_struct.header_size as u64 + res_file_struct.data_size;

    // Open destination file
    let mut out_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(output_file_path)
        .unwrap_or_else(|_| panic!("Cannot create file: {}", output_file_path.to_str().unwrap()));

    // Read file length
    out_file.seek(SeekFrom::End(0)).unwrap();
    let file_len = out_file.stream_position().unwrap();
    out_file.seek(SeekFrom::Start(0)).unwrap();

    // Read file header size
    let file_header_size = if file_len == 0 {
        0
    } else {
        out_file.read_u32::<LittleEndian>().unwrap()
    };
    out_file.seek(SeekFrom::Start(0)).unwrap();

    // Check for updates
    let file_outdated = res_file_struct.header_size != file_header_size || res_file_size != file_len;
    if !res_file_struct.modified && !file_outdated {
        return;
    }

    // Resize file
    out_file
        .set_len(res_file_size)
        .unwrap_or_else(|_| panic!("Cannot resize file: {}", res_file_size));

    // Write header
    out_file
        .write_u32::<LittleEndian>(res_file_struct.header_size)
        .unwrap();
    let mut res_offset = res_file_struct.header_size as u64;

    for entry in &mut res_file_struct.entries {
        if entry.offset == 1 {
            entry.offset = res_offset;
            res_offset += entry.size;
        }

        write!(out_file, "{}\0", entry.name).unwrap();
        out_file.write_u64::<LittleEndian>(entry.offset).unwrap();
        out_file.write_u64::<LittleEndian>(entry.size).unwrap();
    }

    // Write resources
    for entry in &res_file_struct.entries {
        if entry.offset > 0 && (entry.modified || file_outdated) {
            let path_str = entry.path.to_str().unwrap();
            println!("Writing {}", path_str);

            let data = fs::read(entry.path.as_path()).unwrap();
            out_file.seek(SeekFrom::Start(entry.offset)).unwrap();
            out_file.write_all(&data).unwrap();
        }
    }

    // Write timestamp file
    let timestamps = postcard::to_stdvec(&timestamps).unwrap();
    fs::write(&ts_path, timestamps).unwrap();

    println!(
        "Header size: {} B, data size: {} B",
        res_file_struct.header_size, res_file_struct.data_size
    );
}
