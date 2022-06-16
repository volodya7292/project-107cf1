RWByteAddressBuffer mem : register(u0);

#define MORTON_LBVH_GEN
#include "morton.hlsli"

struct PushConstants {
    uint morton_codes_offset;
    uint top_nodes_offset;
    uint instances_offset;
    uint n_leaves;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadId) {
    uint idx = DTid.x;
    uint max_nodes = params.n_leaves * 2 - 1;// `n_elements` leafs and `n_elements - 1` internal nodes.

    if (idx >= max_nodes)
        return;

    SubGlobalBuffer<MortonCode> morton_codes = {mem, params.morton_codes_offset};
    SubGlobalBuffer<LBVHInstance> instances = {mem, params.instances_offset};
    SubGlobalBuffer<TopLBVHNode> nodes = {mem, params.top_nodes_offset};

    if (idx >= params.n_leaves - 1) {
        uint leaf_idx = idx - (params.n_leaves - 1);

        // Leaf node
        uint instance_id = morton_codes.Load(leaf_idx).element_id;
        LBVHInstance instance = instances.Load(instance_id);

        // Store instance data inside TopLBVHNode
        nodes.StoreWithOffset(idx, TopLBVHNode_instance_offset, instance);
    } else {
        // Internal node
        uint2 range = mortonDetermineRange(idx, params.n_leaves, params.morton_codes_offset);
        uint split = mortonFindSplit(range[0], range[1], params.n_leaves, params.morton_codes_offset);
        uint child_a = split;
        uint child_b = split + 1;

        if (child_a == range[0]) {
            // child is a leaf node
            child_a += params.n_leaves - 1;
        }
        if (child_b == range[1]) {
            // child is a leaf node
            child_b += params.n_leaves - 1;
        }

        // Internal node doesn't have inner data
        nodes.StoreWithOffset<uint>(idx, TopLBVHNode_instance_offset + LBVHInstance_nodes_offset_offset, -1);

        // Store child ids
        nodes.StoreWithOffset(idx, TopLBVHNode_child_a_offset, child_a);
        nodes.StoreWithOffset(idx, TopLBVHNode_child_b_offset, child_b);

        // Store parent ids of children
        nodes.StoreWithOffset(child_a, TopLBVHNode_parent_offset, idx);
        nodes.StoreWithOffset(child_b, TopLBVHNode_parent_offset, idx);
    }

    if (idx == 0) {
        // Root node doesn't have parent
        nodes.StoreWithOffset<uint>(0, TopLBVHNode_parent_offset, -1);
    }
}
