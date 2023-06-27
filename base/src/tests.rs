use crate::main_registry::MainRegistry;
use crate::overworld::block_model::Quad;
use crate::overworld::facing::Facing;
use crate::overworld::position::{ClusterBlockPos, RelativeBlockPos};
use crate::overworld::raw_cluster::{deserialize_cluster, serialize_cluster, BlockDataImpl, RawCluster};
use crate::persistence;
use approx::assert_abs_diff_eq;
use common::glm;
use common::glm::{I32Vec3, Vec3};

#[test]
fn cluster_serialization() {
    let main_registry = MainRegistry::init();

    let mut cluster = RawCluster::new();

    cluster.get_mut(&ClusterBlockPos::new(1, 4, 9)).set({
        let mut s = main_registry.block_test_stateful.clone();
        s.components.go = 3;
        s.components.go2 = 2353;
        s
    });

    let mut data = vec![];
    let mut ser = bincode::Serializer::new(&mut data, *persistence::BINCODE_OPTIONS);
    serialize_cluster(&cluster, main_registry.registry(), &mut ser).unwrap();
    assert!(data.len() > 0);

    let mut deser = bincode::Deserializer::from_slice(&data, *persistence::BINCODE_OPTIONS);
    let cluster = deserialize_cluster(main_registry.registry(), &mut deser).unwrap();

    let b_data = cluster.get(&ClusterBlockPos::new(1, 4, 9));
    assert_eq!(*b_data.get::<u32>().unwrap(), 3);
    assert_eq!(*b_data.get::<u64>().unwrap(), 2353)
}

#[test]
fn cluster_facing_routines() {
    assert!(Facing::from_direction(&I32Vec3::new(-1, 0, 1)).is_none());
    assert!(Facing::from_direction(&I32Vec3::new(1, 0, 1)).is_none());
    assert!(Facing::from_direction(&I32Vec3::new(0, 0, 0)).is_none());

    assert_eq!(
        Facing::from_direction(&I32Vec3::new(-1, 0, 0)).unwrap(),
        Facing::NegativeX
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(1, 0, 0)).unwrap(),
        Facing::PositiveX
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, -1, 0)).unwrap(),
        Facing::NegativeY
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 1, 0)).unwrap(),
        Facing::PositiveY
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 0, -1)).unwrap(),
        Facing::NegativeZ
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 0, 1)).unwrap(),
        Facing::PositiveZ
    );

    assert_eq!(Facing::NegativeX.mirror(), Facing::PositiveX);
    assert_eq!(Facing::PositiveX.mirror(), Facing::NegativeX);
    assert_eq!(Facing::NegativeY.mirror(), Facing::PositiveY);
    assert_eq!(Facing::PositiveY.mirror(), Facing::NegativeY);
    assert_eq!(Facing::NegativeZ.mirror(), Facing::PositiveZ);
    assert_eq!(Facing::PositiveZ.mirror(), Facing::NegativeZ);
}

#[test]
fn block_quad_area_calculation() {
    let quad = Quad::new([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ]);

    assert_abs_diff_eq!(quad.area(), 1.0, epsilon = 1e-7);
}

#[test]
fn relative_block_pos_works() {
    let pos = RelativeBlockPos(glm::vec3(-1, 0, 24));
    assert_eq!(pos.cluster_idx(), 5);
    assert_eq!(
        pos.cluster_block_pos(),
        ClusterBlockPos::from_vec_unchecked(glm::vec3(23, 0, 0))
    );
}
