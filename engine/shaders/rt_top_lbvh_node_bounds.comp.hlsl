#include "common.hlsli"

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint top_nodes_offset;
    uint n_elements;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint element_id = DTid.x;
    if (element_id >= params.n_elements) {
        return;
    }

    SubGlobalBuffer<TopLBVHNode> nodes = {buffer, params.top_nodes_offset};

    uint leaf_id = element_id;
    uint parent_id = nodes.Load(leaf_id).parent;

    [loop]
    while (parent_id != -1) {
        TopLBVHNode parent = nodes.Load(parent_id);

        // To avoid duplicate work, use only first child to traverse upwards.
        // It can be done because child_a and child_b are both valid or = -1
        if (leaf_id != parent.child_a) {
            break;
        }

        Bounds bbl = nodes.Load(parent.child_a).instance.bounds;
		Bounds bbr = nodes.Load(parent.child_b).instance.bounds;

        Bounds parent_bounds;
		parent_bounds.p_min = min(bbl.p_min, bbr.p_min);
		parent_bounds.p_max = max(bbl.p_max, bbr.p_max);

		nodes.StoreWithOffset(parent_id, TopLBVHNode_instance_offset + LBVHInstance_bounds_offset, parent_bounds);

		leaf_id = parent_id;
        parent_id = parent.parent;
    }
}