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
} vs_out;

layout(push_constant) uniform PushConstants {
    float pxRange;
    mat3 transform;
    mat4 projView;
} params;

void main() {
    vs_out.texCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vs_out.glyphIndex = inGlyphIndex;
    vs_out.color = inColor;

    vec3 worldPos = params.transform * vec3(vs_out.texCoord*2-1 + inOffset, 0.0f);

    gl_Position = params.projView * vec4(worldPos, 1.0);
//    gl_Position.y = -gl_Position.y;
}