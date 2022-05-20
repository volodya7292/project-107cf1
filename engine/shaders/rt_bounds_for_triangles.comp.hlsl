#include "common.hlsli"

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint indices_offset;
    uint vertices_offset;
    uint nodes_leaves_offset;
    uint n_triangles;
};

static const uint INDEX_SIZE = sizeof(uint);
static const uint VERTEX_SIZE = sizeof(float3);

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
    bounds.p_min = min(min(v0, v1), v2);
    bounds.p_max = max(max(v0, v1), v2);

    buffer.Store(params.nodes_leaves_offset + triangle_id * sizeof(LBVHNode) 
        + LBVHNode_bounds_offset, bounds);
}