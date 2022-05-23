#include "common.hlsli"

// Computes bounds for each transformed LBVH instance

RWByteAddressBuffer buffer : register(u0);

struct PushConstants {
    uint instances_offset;
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

    LBVHInstance instance = buffer.Load<LBVHInstance>(params.instances_offset + element_id * sizeof(LBVHInstance));
    Bounds bb = instance.bounds;

    float3 vertices[8] = {
		float3(bb.p_min.x, bb.p_min.y, bb.p_min.z),
		float3(bb.p_min.x, bb.p_min.y, bb.p_max.z),
		float3(bb.p_min.x, bb.p_max.y, bb.p_min.z),
		float3(bb.p_min.x, bb.p_max.y, bb.p_max.z),
		float3(bb.p_max.x, bb.p_min.y, bb.p_min.z),
		float3(bb.p_max.x, bb.p_min.y, bb.p_max.z),
		float3(bb.p_max.x, bb.p_max.y, bb.p_min.z),
		float3(bb.p_max.x, bb.p_max.y, bb.p_max.z),
	};

    Bounds bounds;
    bounds.p_min = FLT_MAX;
    bounds.p_max = -FLT_MAX;

    [unroll]
	for (uint i = 0; i < 8; i++) {
		vertices[i] = mul(instance.transform, float4(vertices[i], 1)).xyz;
		bounds.p_min = min(bounds.p_min, vertices[i]);
		bounds.p_max = max(bounds.p_max, vertices[i]);
	}

    // Store bounds of the LBVH instance
    buffer.Store(params.instances_offset + element_id * sizeof(LBVHInstance) + LBVHInstance_bounds_offset, bounds);
}