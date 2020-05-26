use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{env, fmt, fs};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

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

    return res_file;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("\nUsage: utxResEncoder.exe <resources directory> <output file>\n");
        return;
    }

    let res_path = Path::new(&args[1]);
    let out_path = Path::new(&args[2]);
    let out_dir_path = Path::new(&args[2]).parent().unwrap();
    let mut ts_path = env::temp_dir();
    ts_path.push("utx_res_encoder_ts");

    if !res_path.exists() {
        println!(
            "Resources directory does not exist: {}",
            res_path.to_str().unwrap()
        );
    }

    // Read timestamp file
    let mut timestamps;
    match fs::read(&ts_path) {
        Ok(x) => match bincode::deserialize(&x) {
            Ok(x) => timestamps = x,
            Err(_) => timestamps = HashMap::new(),
        },
        Err(_) => timestamps = HashMap::new(),
    };

    // Create destination directories
    if !out_dir_path.exists() {
        fs::create_dir_all(out_dir_path)
            .expect(format!("Cannot create output directory: {}", out_path.to_str().unwrap()).as_str());
    }

    // Read resources
    let mut res_file_struct = read_resources(res_path.read_dir().unwrap(), &mut timestamps);
    println!("{:#?}", &res_file_struct);
    res_file_struct.header_size += 4; // + sizeof(u32) for header size var
    let res_file_size = res_file_struct.header_size as u64 + res_file_struct.data_size;
    let file_outdated;

    // Open destination file
    let mut out_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(out_path)
        .expect(format!("Cannot create file: {}", out_path.to_str().unwrap()).as_str());

    // Read file length
    out_file.seek(SeekFrom::End(0)).unwrap();
    let file_len = out_file.seek(SeekFrom::Current(0)).unwrap();
    out_file.seek(SeekFrom::Start(0)).unwrap();

    // Read file header size
    let file_header_size = if file_len == 0 {
        0
    } else {
        out_file.read_u32::<LittleEndian>().unwrap()
    };
    out_file.seek(SeekFrom::Start(0)).unwrap();

    // Check for updates
    file_outdated = res_file_struct.header_size != file_header_size || res_file_size != file_len;
    if !res_file_struct.modified && !file_outdated {
        return;
    }

    // Resize file
    out_file
        .set_len(res_file_size)
        .expect(format!("Cannot resize file: {}", res_file_size).as_str());

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
    fs::write(&ts_path, bincode::serialize(&timestamps).unwrap()).unwrap();

    println!(
        "Header size: {} B, data size: {} B",
        res_file_struct.header_size, res_file_struct.data_size
    );
}
