RWByteAddressBuffer mem : register(u0);

#define MORTON_LBVH_GEN
#include "morton.hlsli"

struct PushConstants {
    uint morton_codes_offset;
    uint leaves_bounds_offset;
    uint nodes_offset;
    uint n_triangles;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadId) {
    uint idx = DTid.x;
    uint max_nodes = params.n_triangles * 2 - 1;// `n_elements` leafs and `n_elements - 1` internal nodes.

    if (idx >= max_nodes)
        return;

    SubGlobalBuffer<MortonCode> morton_codes = {mem, params.morton_codes_offset};
    SubGlobalBuffer<LBVHNode> nodes = {mem, params.nodes_offset};
    SubGlobalBuffer<Bounds> leaves_bounds = {mem, params.leaves_bounds_offset};

    if (idx >= params.n_triangles - 1) {
        // Leaf node
        uint leaf_idx = idx - (params.n_triangles - 1);
        uint element_id = morton_codes.Load(leaf_idx).element_id;
        Bounds bounds = leaves_bounds.Load(element_id);

        // Store element_id
        nodes.StoreWithOffset(idx, LBVHNode_element_id_offset, element_id);
        // Store bounds
        nodes.StoreWithOffset(idx, LBVHNode_bounds_offset, bounds);
    } else {
        // Internal node
        uint2 range = mortonDetermineRange(idx, params.n_triangles, params.morton_codes_offset);
        uint split = mortonFindSplit(range[0], range[1], params.n_triangles, params.morton_codes_offset);
        uint child_a = split;
        uint child_b = split + 1;

        if (child_a == range[0]) {
            // child is a leaf node
            child_a += params.n_triangles - 1;
        }
        if (child_b == range[1]) {
            // child is a leaf node
            child_b += params.n_triangles - 1;
        }

        // Internal node doesn't have inner data
        nodes.StoreWithOffset<uint>(idx, LBVHNode_element_id_offset, -1);

        // Store child ids
        nodes.StoreWithOffset(idx, LBVHNode_child_a_offset, child_a);
        nodes.StoreWithOffset(idx, LBVHNode_child_b_offset, child_b);

        // Store parent ids of children
        nodes.StoreWithOffset(child_a, LBVHNode_parent_offset, idx);
        nodes.StoreWithOffset(child_b, LBVHNode_parent_offset, idx);

        // Bounds _b = { (-FLT_MAX).xxx, FLT_MAX.xxx };
        // nodes.StoreWithOffset(idx, LBVHNode_bounds_offset, _b);
    }

    if (idx == 0) {
        // Root node doesn't have parent
        nodes.StoreWithOffset<uint>(0, LBVHNode_parent_offset, -1);
    }
}
