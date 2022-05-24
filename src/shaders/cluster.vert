#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

#define CLUSTER_SIZE 24

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uvec4 inPack0;

layout(set = 0, binding = 0) uniform per_frame_data {
    PerFrameInfo info;
};
layout(set = 1, binding = 0) uniform per_object_data {
    mat4 model;
};

layout(location = 0) out Output {
    vec2 tex_uv;
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    uint material_id;
} vs_out;

void main() {
    vec3 inNormal = vec3(uvec3((inPack0[0] >> 24) & 0xffu, (inPack0[0] >> 16) & 0xffu, (inPack0[0] >> 8) & 0xffu));
    inNormal = inNormal / 255.0 * 2.0 - 1.0;

    vec2 inTexUV = vec2(uvec2((inPack0[1] >> 16) & 0xffffu, inPack0[1] & 0xffffu));
    inTexUV = inTexUV / 65535.0 * 64.0;

    uint inMaterialId = inPack0[2] & 0xffffu;

    vec4 world_pos = (model * vec4(inPosition, 1));

    vs_out.tex_uv = inTexUV;
    vs_out.local_pos = inPosition;
    vs_out.world_pos = world_pos.xyz;
    vs_out.surface_normal = inNormal;
    vs_out.material_id = inMaterialId;
    gl_Position = info.camera.proj_view * world_pos;
}