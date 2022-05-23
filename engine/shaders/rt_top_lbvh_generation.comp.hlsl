RWByteAddressBuffer mem : register(u0);

#include "common.hlsli"
#define MORTON_LBVH_GEN
#include "morton.hlsli"

struct PushConstants {
    uint morton_codes_offset;
    uint top_nodes_offset;
    uint instances_offset;
    uint n_elements;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadId) {
    uint idx = DTid.x;
    uint max_nodes = params.n_elements * 2 - 1;// `n_elements` leafs and `n_elements - 1` internal nodes.

    if (idx >= max_nodes)
        return;

    TopLBVHNode node = mem.Load<TopLBVHNode>(params.top_nodes_offset + idx * sizeof(TopLBVHNode));

    if (idx < params.n_elements) {
        // Leaf node
        uint instance_id = mem.Load<MortonCode>(params.morton_codes_offset + idx * sizeof(MortonCode)).element_id;
        node.instance = mem.Load<LBVHInstance>(params.instances_offset + instance_id * sizeof(LBVHInstance));
    } else {
        // Internal node
        uint internal_idx = idx - params.n_elements;
        uint2 range = mortonDetermineRange(internal_idx, params.morton_codes_offset, params.n_elements);
        uint split = mortonFindSplit(range[0], range[1], params.morton_codes_offset);

        if (split == range[0]) {
            // child is a leaf node
            node.child_a = split;
        } else {
            // child is an internal node
            node.child_a = params.n_elements + split;
        }

        if (split + 1 == range[1]) {
            // child is a leaf node
            node.child_b = split + 1;
        } else {
            // child is an internal node
            node.child_b = params.n_elements + split + 1;
        }

        node.instance.nodes_offset = -1;
    }

    mem.Store(params.top_nodes_offset + idx * sizeof(TopLBVHNode), node);
}
