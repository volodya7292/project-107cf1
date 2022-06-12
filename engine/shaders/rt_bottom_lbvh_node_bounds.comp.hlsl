#include "common.hlsli"

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint nodes_offset;
    uint atomic_counters_offset;
    uint n_elements;
};

[[vk::push_constant]]
PushConstants params;

// Note: atomics are used because globallycoherent modifier isn't not working on Metal

Bounds LoadBounds(in SubGlobalBuffer<LBVHNode> nodes, uint node_id) {
    Bounds b;
    b.p_min.x = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset));
    b.p_min.y = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset + 4));
    b.p_min.z = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset + 8));
    b.p_max.x = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset + 12));
    b.p_max.y = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset + 16));
    b.p_max.z = asfloat(nodes.AtomicLoad(node_id, LBVHNode_bounds_offset + 20));
    return b;
}

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint element_id = DTid.x;

    if (element_id >= params.n_elements)
        return;

    SubGlobalBuffer<LBVHNode> nodes = {buffer, params.nodes_offset};
    SubGlobalBuffer<uint> atomics = {buffer, params.atomic_counters_offset};

    uint leaf_id = params.n_elements - 1 + element_id;
    uint parent_id = nodes.Load(leaf_id).parent;

    [loop]
    while (parent_id != -1) {
        // Parent AABB is ready to be computed only when its `processed_count` = 2 (two children nodes are ready)
        uint processed_count = atomics.AtomicAdd(parent_id, 0, 1);
        if (processed_count == 0) {
            break;
        }

        LBVHNode parent = nodes.Load(parent_id);

        Bounds bbl = LoadBounds(nodes, parent.child_a);
		Bounds bbr = LoadBounds(nodes, parent.child_b);
        Bounds parent_bounds = bbl.combine(bbr);

		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset, asuint(parent_bounds.p_min.x));
		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset + 4, asuint(parent_bounds.p_min.y));
		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset + 8, asuint(parent_bounds.p_min.z));
		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset + 12, asuint(parent_bounds.p_max.x));
		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset + 16, asuint(parent_bounds.p_max.y));
		nodes.AtomicExchange(parent_id, LBVHNode_bounds_offset + 20, asuint(parent_bounds.p_max.z));
    
        parent_id = parent.parent;
    }
}