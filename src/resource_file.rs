use io::{Seek, SeekFrom};
use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::{fmt, io};

use byteorder::{LittleEndian, ReadBytesExt};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Utf8Error(std::str::Utf8Error),
    InvalidHeader,
    EndOfHeader,
    ResourceNotFound,
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(err: std::str::Utf8Error) -> Self {
        Error::Utf8Error(err)
    }
}

struct ResourceEntry {
    offset: u64,
    size: u64,
    data: Vec<u8>,
    entries: HashMap<String, ResourceEntry>,
}

impl fmt::Debug for ResourceEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceEntry")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("data_size", &self.data.len())
            .field("entries", &self.entries)
            .finish()
    }
}

pub struct ResourceFile {
    buf_reader: Mutex<BufReader<File>>,
    main_entry: ResourceEntry,
}

impl ResourceFile {
    fn read_entries(
        buf_reader: &mut BufReader<File>,
        header_size: u32,
        res_entry: &mut ResourceEntry,
    ) -> Result<(), Error> {
        let curr_pos = buf_reader.seek(SeekFrom::Current(0))?;
        if curr_pos >= header_size as u64 {
            if curr_pos > header_size as u64 {
                return Err(Error::InvalidHeader);
            }
            return Err(Error::EndOfHeader);
        }

        let mut name = Vec::<u8>::new();
        buf_reader.read_until(0, &mut name)?;
        let name = unsafe { CStr::from_bytes_with_nul_unchecked(&name) }.to_str()?;

        let offset = buf_reader.read_u64::<LittleEndian>()?;
        let size = buf_reader.read_u64::<LittleEndian>()?;

        let mut entry = ResourceEntry {
            offset,
            size,
            data: vec![],
            entries: HashMap::new(),
        };

        if entry.offset == 0 {
            // dir
            entry.entries.reserve(size as usize);
            for _ in 0..size {
                Self::read_entries(buf_reader, header_size, &mut entry)?;
            }
        }

        res_entry.entries.insert(name.to_owned(), entry);
        Ok(())
    }

    fn read_entries_full(
        buf_reader: &mut BufReader<File>,
        header_size: u32,
        res_entry: &mut ResourceEntry,
    ) -> Result<(), Error> {
        loop {
            match Self::read_entries(buf_reader, header_size, res_entry) {
                Err(Error::EndOfHeader) => break,
                Err(a) => return Err(a),
                _ => {}
            }
        }
        res_entry.size = res_entry.entries.len() as u64;

        Ok(())
    }

    pub fn open(path: &Path) -> Result<Arc<ResourceFile>, Error> {
        let mut res_file = ResourceFile {
            buf_reader: Mutex::new(BufReader::new(File::open(path)?)),
            main_entry: ResourceEntry {
                offset: 0,
                size: 0,
                data: vec![],
                entries: HashMap::new(),
            },
        };
        {
            let mut buf_reader = res_file.buf_reader.lock().unwrap();
            let header_size = buf_reader.read_u32::<LittleEndian>()?;
            Self::read_entries_full(&mut buf_reader, header_size, &mut res_file.main_entry)?;
        }
        Ok(Arc::new(res_file))
    }

    fn get_res_range(&self, filename: &str) -> Result<(u64, u64), Error> {
        let mut entry = &self.main_entry;

        for name in filename.split('/') {
            match entry.entries.get(name) {
                Some(a) => entry = a,
                _ => return Err(Error::ResourceNotFound),
            }
        }

        return Ok((entry.offset, entry.size));
    }

    fn read_range(&self, range: (u64, u64)) -> Result<Vec<u8>, Error> {
        let mut buf_reader = self.buf_reader.lock().unwrap();

        buf_reader.seek(SeekFrom::Start(range.0))?;
        let mut data = vec![0u8; range.1 as usize];
        buf_reader.read(&mut data)?;

        Ok(data)
    }

    pub fn get(self: &Arc<Self>, filename: &str) -> Result<ResourceRef, Error> {
        let range = self.get_res_range(filename)?;

        Ok(ResourceRef {
            res_file: Arc::clone(self),
            range,
        })
    }
}

#[derive(Clone)]
pub struct ResourceRef {
    res_file: Arc<ResourceFile>,
    range: (u64, u64),
}

impl ResourceRef {
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        self.res_file.read_range(self.range)
    }
}
