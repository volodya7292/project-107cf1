use crate::object::cluster;

pub trait WorldGenerator {
    fn generate_cluster(&self) -> Vec<cluster::DensityPointInfo>;
}
