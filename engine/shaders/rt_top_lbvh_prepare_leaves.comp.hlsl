#include "common.hlsli"
#include "morton.hlsli"

// Computes morton code for each TopLBVHNode in the scene

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint morton_codes_offset;
    uint top_nodes_offset;
    uint n_elements;
    float3 scene_bound_min;
    float3 scene_bound_max;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint element_id = DTid.x;
    if (element_id >= params.n_elements) {
        return;
    }

    // Leaves go after internal nodes. 
    // The number of internal nodes are one less than the number of leaves.
    uint nodes_leaves_offset = params.top_nodes_offset + (params.n_elements - 1) * sizeof(TopLBVHNode); 

    Bounds bounds = buffer.Load<TopLBVHNode>(nodes_leaves_offset + element_id * sizeof(TopLBVHNode)).bounds;

    float3 center = aabbComputeCenter(params.scene_bound_min, params.scene_bound_max, bounds.p_min, bounds.p_max);
    uint code = morton3D(center);

    MortonCode morton_code;
    morton_code.code = code;
    morton_code.element_id = element_id;

    // Store triangle's morton code 
    buffer.Store(params.morton_codes_offset + element_id * sizeof(MortonCode), morton_code);
}