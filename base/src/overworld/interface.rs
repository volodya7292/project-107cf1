pub mod block_states;
pub mod local_interface;

use crate::overworld::generator::OverworldGenerator;
use crate::overworld::position::ClusterPos;
use crate::overworld::raw_cluster::RawCluster;
use crate::overworld::ClusterState;
use common::parking_lot::RwLock;
use lazy_static::lazy_static;
use std::sync::Arc;

lazy_static! {
    pub static ref BINCODE_OPTIONS: bincode::DefaultOptions = bincode::options();
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum LoadedType {
    NewlyGenerated,
    Loaded,
}

pub trait OverworldInterface: Send + Sync {
    /// Immediately loads cluster from the underlying storage.
    fn load_cluster(&self, pos: &ClusterPos) -> (RawCluster, LoadedType);
    /// Queues the cluster for saving into the underlying storage.
    fn persist_cluster(&self, pos: &ClusterPos, cluster: Arc<RwLock<ClusterState>>);
    fn generator(&self) -> &Arc<OverworldGenerator>;
}
