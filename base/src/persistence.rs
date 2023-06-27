pub mod block_states;

use lazy_static::lazy_static;

lazy_static! {
    pub static ref BINCODE_OPTIONS: bincode::DefaultOptions = bincode::options();
}
