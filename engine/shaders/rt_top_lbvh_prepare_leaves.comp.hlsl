#include "morton.hlsli"

// Computes morton code for each instance in the scene

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint morton_codes_offset;
    uint instances_offset;
    uint scene_bounds_offset;
    uint n_leaves;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint element_id = DTid.x;
    if (element_id >= params.n_leaves) {
        return;
    }

    Bounds scene_bounds = buffer.Load<Bounds>(params.scene_bounds_offset);
    Bounds bounds = buffer.Load<LBVHInstance>(params.instances_offset + element_id * sizeof(LBVHInstance)).bounds;

    float3 center = aabbComputeCenter(scene_bounds.p_min, scene_bounds.p_max, bounds.p_min, bounds.p_max);
    uint code = morton3D(center);

    MortonCode morton_code;
    morton_code.code = code;
    morton_code.element_id = element_id;

    // Store triangle's morton code 
    buffer.Store(params.morton_codes_offset + element_id * sizeof(MortonCode), morton_code);
}