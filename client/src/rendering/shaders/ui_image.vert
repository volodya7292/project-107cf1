#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../../engine/shaders/common.glsl"
#include "ui.glsl"

layout(set = 0, binding = 0, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    vec2 img_offset;
    vec2 img_scale;
};

layout(location = 0) out Output {
    vec2 texCoord;
} vs_out;

void main() {
    vec2 pos2d = vec2(gl_VertexIndex >> 1, gl_VertexIndex & 1);

    vec2 texCoord = vec2(pos2d.x, 1.0 - pos2d.y);
    texCoord -= img_offset;
    texCoord /= img_scale;

    vec4 worldPos = model * vec4(pos2d, 0.0, 1.0);
    vs_out.texCoord = texCoord;
    gl_Position = info.camera.proj_view * worldPos;
}
