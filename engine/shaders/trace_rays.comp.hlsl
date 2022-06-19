RWByteAddressBuffer mem : register(u0);
#include "ray_tracing.hlsli"

RWTexture2D<float4> output : register(u1);

struct PushConstants {
    float2 resolution;
    uint top_nodes_offset;
};

ConstantBuffer<PerFrameInfo> frame_info : register(b2);

[[vk::push_constant]]
PushConstants params;

void primary_ray(uint2 screen_size, float2 screen_pixel_pos, out float3 ray_orig, out float3 ray_dir) {
	const float ct = tan((60.0 * 3.14 / 180.0) / 2.0f);
	const float2 screenPos = (screen_pixel_pos / (float2)screen_size * 2.0f - 1.0f) * float2(ct * ((float)screen_size.x / screen_size.y), -ct);

	ray_orig = frame_info.camera.pos.xyz;
	ray_dir = mul(float4(normalize(float3(screenPos, -1)), 1.0), frame_info.camera.view).xyz;
    // ray_dir = -ray_dir;
}

[numthreads(THREAD_GROUP_WIDTH, THREAD_GROUP_HEIGHT, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (any(DTid.xy > params.resolution)) {
        return;
    }

    float3 ray_orig, ray_dir;
    primary_ray(params.resolution, DTid.xy, ray_orig, ray_dir);

    TriangleIntersection inter = trace_ray(ray_orig, ray_dir, params.top_nodes_offset);
    // float4 a = mem.Load<float4>(0);

    // rt_top_traversal_stack[uint(params.resolution.x)] = 3;



    float4 a = inter.intersected ? float4(inter.inter_point + 0.5, 1) : float4(0, 0, 0, 1);

    // output[DTid.xy] = float4(ray_dir, 1) * clamp(a, 1.0, 1.0);
    output[DTid.xy] = a;
}