#include "common.hlsli"

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint indices_offset;
    uint vertices_offset;
    uint morton_codes_offset;
    uint n_triangles;
};

static const float ONE_THIRD = 1.0 / 3.0;
static const uint INDEX_SIZE = sizeof(uint);
static const uint VERTEX_SIZE = sizeof(float3);

[[vk::push_constant]]
PushConstants params;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
uint3 expandBits(uint3 v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(float3 p) {
    uint3 upoint = uint3(clamp(p + 512.0, 0.0, 1023.0));
    uint3 ex = expandBits(upoint);
    return ex.x * 4 + ex.y * 2 + ex.z;
}

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

    float3 center = (v0 + v1 + v2) * ONE_THIRD;
    uint code = morton3D(center);

    MortonCode morton_code;
    morton_code.code = code;
    morton_code.element_id = triangle_id;

    buffer.Store(params.morton_codes_offset + triangle_id * sizeof(MortonCode), morton_code);
}