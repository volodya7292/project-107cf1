#include "morton.hlsli"

// Computes bounds and morton code for each triangle in a mesh

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint indices_offset;
    uint vertices_offset;
    uint morton_codes_offset;
    uint nodes_offset;
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

    LBVHNode node;
    node.bounds.p_min = min(v0, min(v1, v2));
    node.bounds.p_max = max(v0, max(v1, v2));
    node.element_id = -1;
    node.parent = -1;
    node.child_a = -1;
    node.child_b = -1;

    float3 center = aabbComputeCenter(params.mesh_bound_min, params.mesh_bound_max,
        node.bounds.p_min, node.bounds.p_max);
    uint code = morton3D(center);

    MortonCode morton_code;
    morton_code.code = code;
    morton_code.element_id = triangle_id;

    uint leaves_offset = params.nodes_offset + (params.n_triangles - 1) * sizeof(LBVHNode);

    // Store triangle's prepared node
    buffer.Store(leaves_offset + triangle_id * sizeof(LBVHNode), node);
    // Store triangle's morton code 
    buffer.Store(params.morton_codes_offset + triangle_id * sizeof(MortonCode), morton_code);
}