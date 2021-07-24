#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform per_frame_data {
    PerFrameInfo info;
};
layout(set = 1, binding = 0) uniform per_object_data {
    mat4 model;
};

layout(location = 0) out vec2 tex_coord;

void main() {
    tex_coord = inTexCoord;
    gl_Position = info.camera.proj_view * (model * vec4(inPosition, 1));
}