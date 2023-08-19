pub mod block_states;
pub mod local_interface;

use crate::overworld::generator::OverworldGenerator;
use crate::overworld::position::ClusterPos;
use crate::overworld::raw_cluster::RawCluster;
use crate::overworld::{ClusterState, OverworldState};
use common::parking_lot::RwLock;
use std::any::Any;
use std::sync::Arc;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum LoadedType {
    NewlyGenerated,
    Loaded,
}

pub trait OverworldInterface: Send + Sync {
    fn generator(&self) -> &Arc<OverworldGenerator>;

    /// Immediately loads cluster from the underlying storage.
    fn load_cluster(&self, pos: &ClusterPos) -> (RawCluster, LoadedType);
    /// Queues the cluster for saving into the underlying storage.
    fn persist_cluster(&self, pos: &ClusterPos, cluster: Arc<RwLock<ClusterState>>);

    fn persisted_state(&self) -> OverworldState;
    fn persist_state(&self, params: OverworldState);

    fn as_any(&self) -> &dyn Any;
}
