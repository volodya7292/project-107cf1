RWByteAddressBuffer mem : register(u0);
#include "ray_tracing.hlsli"

RWTexture2D<float4> output : register(u1);

struct PushConstants {
    float2 resolution;
    uint top_nodes_offset;
};

[[vk::push_constant]]
PushConstants params;

[numthreads(THREAD_GROUP_WIDTH, THREAD_GROUP_HEIGHT, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (any(DTid.xy > params.resolution)) {
        return;
    }
    rt_top_nodes_offset = params.top_nodes_offset;

    float4 a = mem.Load<float4>(0);

    output[DTid.xy] = clamp(a, 1.0, 1.0);
}