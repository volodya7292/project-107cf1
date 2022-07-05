#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(location = 0) in uint inGlyphIndex;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inOffset;

layout(location = 0) out Output {
    vec2 texCoord;
    uint glyphIndex;
    vec4 color;
    float pxRange;
} vs_out;

layout(std140, set = 0, binding = 0) uniform UniformData {
    float pxRange;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    mat4 projView;
};

layout(push_constant) uniform PushConstants {
    mat4 transform;
} params;

void main() {
    vs_out.texCoord = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
    vs_out.glyphIndex = inGlyphIndex;
    vs_out.color = inColor;
    vs_out.pxRange = pxRange;

    vec4 worldPos = params.transform * vec4(vs_out.texCoord*2-1 + inOffset, 0.0, 1.0);

    gl_Position = projView * worldPos;
//    gl_Position.y = -gl_Position.y;
}