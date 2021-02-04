#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform per_frame_data {
    PerFrameInfo info;
};
layout(set = 1, binding = 0) uniform per_object_data {
    mat4 model;
};

layout(location = 0) out Output {
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    vec2 tex_coord;
} vs_out;

void main() {
    vec4 world_pos = (model * vec4(inPosition, 1));

    vs_out.local_pos = inPosition;
    vs_out.world_pos = world_pos.xyz;
    vs_out.surface_normal = inNormal;
    vs_out.tex_coord = vec2(0, 0);

    gl_Position = info.camera.proj_view * world_pos;
}