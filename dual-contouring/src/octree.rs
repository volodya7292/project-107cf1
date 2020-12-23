use nalgebra as na;

#[derive(Clone, Debug)]
pub enum NodeType<T> {
    Internal([u32; 8]),
    Leaf(T),
    External(Octree<T>),
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    ty: NodeType<T>,
    size: u32,
}

impl<T> Node<T> {
    pub fn new_leaf(size: u32, data: T) -> Node<T> {
        Node {
            ty: NodeType::Leaf(data),
            size,
        }
    }

    pub fn ty(&self) -> &NodeType<T> {
        &self.ty
    }

    pub fn size(&self) -> u32 {
        self.size
    }
}

#[repr(C)]
union EncodedNodeData<T: Copy> {
    children: [u32; 8],
    leaf: T,
}

pub const ENCODED_TYPE_INTERNAL: u8 = 0;
pub const ENCODED_TYPE_LEAF: u8 = 1;

#[repr(C)]
pub struct EncodedNode<T: Copy> {
    ty: u8,
    data: EncodedNodeData<T>,
}

#[derive(Clone, Debug)]
pub struct Octree<T> {
    size: u32,
    nodes: Vec<Option<Node<T>>>,
    free_node_ids: Vec<u32>,
}

pub(crate) const NODE_ID_NONE: u32 = u32::MAX;

impl<T> Octree<T> {
    fn alloc_node(&mut self) -> u32 {
        if self.free_node_ids.is_empty() {
            self.nodes.push(None);
            (self.nodes.len() - 1) as u32
        } else {
            self.free_node_ids.remove(self.free_node_ids.len() - 1) as u32
        }
    }

    fn remove_node(&mut self, index: u32) {
        let index = index as usize;
        let children;

        // Check for node
        if let Some(node) = &self.nodes[index] {
            children = if let NodeType::Internal(children) = &node.ty {
                Some(*children)
            } else {
                None
            };
        } else {
            return;
        }

        // Remove children nodes
        if let Some(children) = &children {
            for &child_id in children {
                if child_id != NODE_ID_NONE {
                    self.remove_node(child_id);
                }
            }
        }

        // Remove parent node
        if index == (self.nodes.len() - 1) {
            self.nodes.remove(index);
        } else {
            self.nodes[index] = None;
            self.free_node_ids.push(index as u32);
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub(crate) fn get_node(&self, id: u32) -> &Option<Node<T>> {
        &self.nodes[id as usize]
    }

    pub(crate) fn set_node(&mut self, position: na::Vector3<u32>, node: Node<T>) {
        let mut curr_node_id = 0;
        let mut curr_pos = position;
        let mut curr_size = self.size;

        while curr_size != node.size {
            // Check for internal node availability
            {
                let curr_node = &mut self.nodes[curr_node_id as usize];

                if curr_node.is_none() {
                    *curr_node = Some(Node {
                        ty: NodeType::Internal([NODE_ID_NONE; 8]),
                        size: curr_size,
                    });
                }
            }

            // Calculate child index
            let child_index = curr_pos / (curr_size / 2);
            let child_index_1d = (child_index.x * 4 + child_index.y * 2 + child_index.z) as usize;

            curr_size /= 2;
            curr_pos -= curr_size * child_index;

            // Get child node id
            let mut child_id = {
                let curr_node = self.nodes[curr_node_id as usize].as_mut().unwrap();

                match &mut curr_node.ty {
                    NodeType::Internal(children) => children[child_index_1d],
                    NodeType::Leaf(_) => {
                        curr_node.ty = NodeType::Internal([NODE_ID_NONE; 8]);
                        NODE_ID_NONE
                    }
                    NodeType::External(octree) => {
                        octree.set_node(curr_pos, node);
                        return;
                    }
                }
            };

            // Check child node id
            if child_id == NODE_ID_NONE {
                child_id = self.alloc_node();
                let curr_node = self.nodes[curr_node_id as usize].as_mut().unwrap();

                if let NodeType::Internal(children) = &mut curr_node.ty {
                    children[child_index_1d] = child_id;
                }
            }

            curr_node_id = child_id;
        }

        // Get children nodes
        let children = if let Some(curr_node) = &mut self.nodes[curr_node_id as usize] {
            if let NodeType::Internal(children) = curr_node.ty {
                Some(children)
            } else {
                None
            }
        } else {
            None
        };

        // Remove children nodes
        if let Some(children) = children {
            for &child_id in &children {
                if child_id != NODE_ID_NONE {
                    self.remove_node(child_id);
                }
            }
        }

        // Set new node
        self.nodes[curr_node_id as usize] = Some(node);
    }

    fn encode_node<F, M>(&self, map: &F, node_id: u32, buffer: &mut Vec<EncodedNode<M>>) -> u32
    where
        F: Fn(&T) -> M,
        M: Copy,
    {
        if let Some(node) = &self.nodes[node_id as usize] {
            match &node.ty {
                NodeType::Internal(children) => {
                    let mut children = *children;

                    for child in &mut children {
                        if *child != NODE_ID_NONE {
                            *child = self.encode_node(map, *child, buffer);
                        }
                    }

                    buffer.push(EncodedNode {
                        ty: ENCODED_TYPE_INTERNAL,
                        data: EncodedNodeData { children },
                    });
                    buffer.len() as u32 - 1
                }
                NodeType::Leaf(data) => {
                    buffer.push(EncodedNode {
                        ty: ENCODED_TYPE_LEAF,
                        data: EncodedNodeData { leaf: map(data) },
                    });
                    buffer.len() as u32 - 1
                }
                NodeType::External(octree) => octree.encode_node(map, 0, buffer),
            }
        } else {
            NODE_ID_NONE
        }
    }

    /// Converts octree to an array of encoded nodes.
    pub fn encode_into_buffer<F, M>(&self, map: F) -> Vec<EncodedNode<M>>
    where
        F: Fn(&T) -> M,
        M: Copy,
    {
        let mut buffer = Vec::<EncodedNode<M>>::with_capacity(self.nodes.len());
        self.encode_node(&map, 0, &mut buffer);
        buffer
    }
}

/// Create a new octree with specified node capacity.
pub fn with_capacity<T>(size: u32, capacity: u32) -> Octree<T> {
    let mut octree = new(size);
    octree.nodes.reserve(capacity as usize);
    octree.free_node_ids.reserve(capacity as usize);
    octree
}

pub fn new<T>(size: u32) -> Octree<T> {
    Octree {
        size,
        nodes: vec![None],
        free_node_ids: vec![],
    }
}

#[derive(Copy, Clone)]
pub struct LeafNode<T> {
    data: T,
    pos: na::Vector3<u32>,
    size: u32,
}

impl<T> LeafNode<T> {
    pub fn new(pos: na::Vector3<u32>, size: u32, data: T) -> LeafNode<T> {
        LeafNode { data, pos, size }
    }

    pub fn position(&self) -> &na::Vector3<u32> {
        &self.pos
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub fn from_nodes<T>(size: u32, nodes: &[LeafNode<T>]) -> Octree<T>
where
    T: Clone,
{
    let mut octree = with_capacity::<T>(size, (nodes.len() * 2) as u32);

    for node in nodes {
        octree.set_node(
            node.pos,
            Node {
                ty: NodeType::Leaf(node.data.clone()),
                size: node.size,
            },
        )
    }

    octree
}
