pub mod contour;
mod octree;
mod utils;

#[cfg(test)]
mod tests {
    use crate::octree;
    use nalgebra as na;

    #[test]
    fn octree_set_node() {
        let mut oct = octree::new::<u32>(16);

        oct.set_node(na::Vector3::new(11, 11, 11), 1, 123);
        oct.set_node(na::Vector3::new(11, 11, 10), 1, 321);

        println!("{:#?}", oct);
    }
}
