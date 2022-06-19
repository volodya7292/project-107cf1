#include "morton.hlsli"

// Computes bounds and morton code for each triangle in a mesh

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint indices_offset;
    uint vertices_offset;
    uint morton_codes_offset;
    uint nodes_offset;
    uint leaves_bounds_offset;
    uint n_triangles;
    float3 mesh_bound_min;
    float3 mesh_bound_max;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint triangle_id = DTid.x;
    if (triangle_id >= params.n_triangles) {
        return;
    }

    SubGlobalBuffer<uint3> indices = {buffer, params.indices_offset};
    SubGlobalBuffer<float3> vertices = {buffer, params.vertices_offset};
    SubGlobalBuffer<LBVHNode> nodes = {buffer, params.nodes_offset};
    SubGlobalBuffer<Bounds> leaves_bounds = {buffer, params.leaves_bounds_offset};
    SubGlobalBuffer<MortonCode> morton_codes = {buffer, params.morton_codes_offset};

    uint3 tri_indices = indices.Load(triangle_id);
    float3 v0 = vertices.Load(tri_indices[0]);
    float3 v1 = vertices.Load(tri_indices[1]);
    float3 v2 = vertices.Load(tri_indices[2]);

    Bounds bounds;
    bounds.p_min = min(v0, min(v1, v2));
    bounds.p_max = max(v0, max(v1, v2));

    LBVHNode node;
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

    uint leaves_start = params.n_triangles - 1;

    // Store triangle's prepared node
    nodes.Store(leaves_start + triangle_id, node);
    // Store triangle's morton code 
    morton_codes.Store(triangle_id, morton_code);
    // Store bounds
    leaves_bounds.Store(triangle_id, bounds);
}