#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexUV;
layout(location = 3) in float inAO;
layout(location = 4) in uint inMaterialId;

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
    vec4 world_pos = (model * vec4(inPosition, 1));

    vs_out.tex_uv = inTexUV;
    vs_out.local_pos = inPosition;
    vs_out.world_pos = world_pos.xyz;
    vs_out.surface_normal = inNormal;
    vs_out.material_id = inMaterialId;
    gl_Position = info.camera.proj_view * world_pos;
}