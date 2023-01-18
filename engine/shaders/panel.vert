#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(set = 0, binding = 0, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = 1, binding = 0) uniform ObjectData {
    mat4 model;
};

layout (location = 0) out vec2 outUV;

void main() {
    vec2 pos2d = vec2(gl_VertexIndex >> 1, gl_VertexIndex & 1);

    vec4 worldPos = model * vec4(pos2d, 0.0, 1.0);
    gl_Position = info.overlay_camera.proj_view * worldPos;
}
