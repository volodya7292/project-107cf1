#include "common.hlsli"
#include "morton.hlsli"

// Computes bounds and morton code for each triangle in a mesh

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint indices_offset;
    uint vertices_offset;
    uint morton_codes_offset;
    uint leaf_bounds_offset;
    uint n_triangles;
    float3 mesh_bound_min;
    float3 mesh_bound_max;
};

#define INDEX_SIZE sizeof(uint)
#define VERTEX_SIZE sizeof(float3)

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint triangle_id = DTid.x;
    if (triangle_id >= params.n_triangles) {
        return;
    }

    uint first_index_ptr = params.indices_offset + triangle_id * 3 * INDEX_SIZE;
    uint3 v_ptrs = params.vertices_offset.xxx + buffer.Load<uint3>(first_index_ptr) * VERTEX_SIZE;

    float3 v0 = buffer.Load<float3>(v_ptrs[0]);
    float3 v1 = buffer.Load<float3>(v_ptrs[1]);
    float3 v2 = buffer.Load<float3>(v_ptrs[2]);

    Bounds bounds;
    bounds.p_min = min(v0, min(v1, v2));
    bounds.p_max = max(v0, max(v1, v2));


    float3 center = aabbComputeCenter(params.mesh_bound_min, params.mesh_bound_max, bounds.p_min, bounds.p_max);
    uint code = morton3D(center);

    MortonCode morton_code;
    morton_code.code = code;
    morton_code.element_id = triangle_id;


    // Store triangle's bounds
    buffer.Store(params.leaf_bounds_offset + triangle_id * sizeof(Bounds), bounds);
    // Store triangle's morton code 
    buffer.Store(params.morton_codes_offset + triangle_id * sizeof(MortonCode), morton_code);
}