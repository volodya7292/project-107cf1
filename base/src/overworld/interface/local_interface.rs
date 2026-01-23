use crate::execution::default_queue;
use crate::execution::timer::IntervalTimer;
use crate::execution::virtual_processor::VirtualProcessor;
use crate::main_registry::MainRegistry;
use crate::overworld::generator::OverworldGenerator;
use crate::overworld::interface::{LoadedType, OverworldInterface};
use crate::overworld::position::ClusterPos;
use crate::overworld::raw_cluster::{RawCluster, deserialize_cluster, serialize_cluster};
use crate::overworld::{ClusterState, OverworldState};
use crate::registry::Registry;
use common::glm::{I64Vec3, TVec3};
use common::moka;
use common::parking_lot::{Mutex, RwLock};
use common::types::{ConcurrentCache, HashMap};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::{fs, io};

const SECTOR_SIZE_1D: usize = 5;
const SECTOR_SIZE_CELLS_1D: usize = SECTOR_SIZE_1D * RawCluster::SIZE;
const SECTORS_DIR_NAME: &'static str = "matter";
const OVERWORLD_STATE_FILE_NAME: &'static str = "state.json";

lazy_static! {
    pub static ref BINCODE_OPTIONS: bincode::DefaultOptions = bincode::options();
}

fn make_sector_file_name(pos: &SectorPos) -> String {
    let raw_pos = pos.get();
    format!("{}_{}_{}", raw_pos.x, raw_pos.y, raw_pos.z)
}

fn save_state(params: &OverworldState, overworld_folder: &Path) {
    let params_path = overworld_folder.join(OVERWORLD_STATE_FILE_NAME);
    fs::write(params_path, serde_json::to_string(params).unwrap()).unwrap();
}

fn on_commit(
    folder: &Path,
    registry: &Arc<Registry>,
    sectors_cache: &SectorsCache,
    to_save: &Arc<Mutex<HashMap<ClusterPos, Arc<RwLock<ClusterState>>>>>,
    params: &Arc<Mutex<OverworldState>>,
) {
    let clusters: Vec<_> = to_save.lock().drain().collect();
    let mut by_sectors = HashMap::<SectorPos, Arc<RwLock<SectorData>>>::new();

    // Load and modify respective sectors
    for (cluster_pos, cl) in clusters {
        let sector_pos = SectorPos::from_cluster_pos(&cluster_pos);
        let sector_data = by_sectors
            .entry(sector_pos)
            .or_insert_with(|| sectors_cache.get(&sector_pos));

        let ready_cluster = ClusterState::ready(&cl).unwrap();
        let t_cluster = ready_cluster.unwrap();

        let mut out_data = Vec::with_capacity(64_000);
        let mut serializer = bincode::Serializer::new(&mut out_data, *BINCODE_OPTIONS);
        serialize_cluster(&t_cluster.raw, registry, &mut serializer).unwrap();
        let decompressed_size = out_data.len();
        let compressed_data = lz4_flex::compress(&out_data);

        let offset = SectorPos::calc_cluster_idx_offset(&cluster_pos);
        sector_data.write().set_cluster(
            &offset,
            ClusterData {
                compressed_data,
                decompressed_size,
            },
        );
    }

    // Create/replace sector files
    for (sector_pos, sector) in by_sectors {
        let sector = sector.read();
        let sector_bytes = bincode::serialize(&*sector).unwrap();
        let file_name = make_sector_file_name(&sector_pos);
        fs::write(folder.join(SECTORS_DIR_NAME).join(file_name), sector_bytes).unwrap();
    }

    save_state(&params.lock().clone(), folder);
}

pub struct LocalOverworldInterface {
    folder: PathBuf,
    registry: Arc<Registry>,
    generator: Arc<OverworldGenerator>,
    sectors_cache: Arc<SectorsCache>,
    to_save: Arc<Mutex<HashMap<ClusterPos, Arc<RwLock<ClusterState>>>>>,
    params: Arc<Mutex<OverworldState>>,
    save_worker: IntervalTimer,
}

pub struct SectorsCache {
    folder: PathBuf,
    cache: ConcurrentCache<SectorPos, Arc<RwLock<SectorData>>>,
}

impl SectorsCache {
    fn new(folder: PathBuf) -> Self {
        Self {
            folder,
            cache: moka::sync::CacheBuilder::new(256)
                .time_to_idle(Duration::from_secs(10))
                .build_with_hasher(common::types::Hasher::new()),
        }
    }

    fn get(&self, sector_pos: &SectorPos) -> Arc<RwLock<SectorData>> {
        self.cache.get_with(*sector_pos, || {
            let file_name = make_sector_file_name(sector_pos);
            let file = fs::File::open(self.folder.join(SECTORS_DIR_NAME).join(file_name));

            let file = match file {
                Ok(file) => file,
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    return Arc::new(RwLock::new(SectorData::new()));
                }
                err => err.unwrap(),
            };

            let data = bincode::deserialize_from::<_, SectorData>(file).unwrap();
            Arc::new(RwLock::new(data))
        })
    }
}

impl LocalOverworldInterface {
    pub fn create_overworld(folder: impl Into<PathBuf>, params: OverworldState) {
        let folder = folder.into();
        if !folder.exists() {
            fs::create_dir(&folder).unwrap();
        }

        let params_path = folder.join(OVERWORLD_STATE_FILE_NAME);

        fs::write(params_path, serde_json::to_string(&params).unwrap()).unwrap();
    }

    pub fn new(folder: impl Into<PathBuf>, main_registry: &Arc<MainRegistry>) -> Self {
        let folder = folder.into();
        let sectors_dir_path = folder.join(SECTORS_DIR_NAME);
        let params_path = folder.join(OVERWORLD_STATE_FILE_NAME);

        let to_save = Arc::new(Mutex::new(HashMap::new()));
        let sectors_cache = Arc::new(SectorsCache::new(folder.clone()));

        if !sectors_dir_path.exists() {
            fs::create_dir(sectors_dir_path).unwrap();
        }

        let params: OverworldState = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
        let generator = Arc::new(OverworldGenerator::new(params.seed, main_registry));

        let params = Arc::new(Mutex::new(params));
        let save_worker = {
            let folder = folder.clone();
            let sectors_cache = Arc::clone(&sectors_cache);
            let to_save = Arc::clone(&to_save);
            let registry = Arc::clone(generator.main_registry().registry());
            let params = Arc::clone(&params);
            IntervalTimer::start(
                Duration::from_secs(15),
                VirtualProcessor::new(&default_queue().unwrap()),
                move || on_commit(&folder, &registry, &sectors_cache, &to_save, &params),
            )
        };

        Self {
            folder,
            registry: Arc::clone(generator.main_registry().registry()),
            generator,
            sectors_cache,
            to_save,
            params,
            save_worker,
        }
    }

    pub fn stop_and_commit_changes(&self) {
        self.save_worker.stop_and_join();
        on_commit(
            &self.folder,
            &self.registry,
            &self.sectors_cache,
            &self.to_save,
            &self.params,
        );
    }
}

impl OverworldInterface for LocalOverworldInterface {
    fn generator(&self) -> &Arc<OverworldGenerator> {
        &self.generator
    }

    fn load_cluster(&self, pos: &ClusterPos) -> (RawCluster, LoadedType) {
        let sector_pos = SectorPos::from_cluster_pos(pos);
        let file_data = self.sectors_cache.get(&sector_pos);
        let file_data = file_data.read();

        // Load cluster from file if it exists
        let offset = SectorPos::calc_cluster_idx_offset(pos);
        if let Some(cluster_data) = file_data.get_cluster(&offset) {
            return (cluster_data.load(&self.registry), LoadedType::Loaded);
        }

        // Generate the cluster
        let mut cluster = RawCluster::new();
        self.generator.generate_cluster(&mut cluster, *pos);
        (cluster, LoadedType::NewlyGenerated)
    }

    fn persist_cluster(&self, pos: &ClusterPos, cluster: Arc<RwLock<ClusterState>>) {
        self.to_save.lock().insert(*pos, cluster);
    }

    fn persisted_state(&self) -> OverworldState {
        *self.params.lock()
    }

    fn update_persisted_state(&self, f: &dyn Fn(&mut OverworldState)) {
        f(&mut self.params.lock());
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
pub struct ClusterData {
    #[serde(with = "serde_bytes")]
    compressed_data: Vec<u8>,
    decompressed_size: usize,
}

impl ClusterData {
    pub fn load(&self, registry: &Registry) -> RawCluster {
        let bytes = lz4_flex::decompress(&self.compressed_data, self.decompressed_size).unwrap();
        let mut deserializer = bincode::Deserializer::from_slice(&bytes, *BINCODE_OPTIONS);
        return deserialize_cluster(registry, &mut deserializer).unwrap();
    }
}

#[derive(Serialize, Deserialize)]
struct SectorData(Vec<Option<ClusterData>>);

impl SectorData {
    fn new() -> Self {
        Self((0..SECTOR_SIZE_1D.pow(3)).map(|_| None).collect())
    }

    fn offset_to_idx(offset: &TVec3<usize>) -> usize {
        assert!(*offset < TVec3::from_element(SECTOR_SIZE_1D));
        const SECTOR_SIZE_SQR: usize = SECTOR_SIZE_1D * SECTOR_SIZE_1D;
        offset.x * SECTOR_SIZE_SQR + offset.y * SECTOR_SIZE_1D + offset.z
    }

    fn set_cluster(&mut self, offset: &TVec3<usize>, data: ClusterData) {
        self.0[Self::offset_to_idx(offset)] = Some(data);
    }

    fn get_cluster(&self, offset: &TVec3<usize>) -> Option<&ClusterData> {
        self.0.get(Self::offset_to_idx(offset))?.as_ref()
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
struct SectorPos(I64Vec3);

impl SectorPos {
    pub fn from_cluster_pos(cluster_pos: &ClusterPos) -> Self {
        Self(
            cluster_pos
                .get()
                .map(|v| v.div_euclid(SECTOR_SIZE_CELLS_1D as i64))
                * SECTOR_SIZE_CELLS_1D as i64,
        )
    }

    pub fn calc_cluster_idx_offset(cluster_pos: &ClusterPos) -> TVec3<usize> {
        cluster_pos
            .get()
            .map(|v| v.rem_euclid(SECTOR_SIZE_CELLS_1D as i64) as usize)
            / RawCluster::SIZE
    }

    pub fn get(&self) -> &I64Vec3 {
        &self.0
    }
}
