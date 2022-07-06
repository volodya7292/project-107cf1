#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(location = 0) in uint inGlyphIndex;
layout(location = 1) in vec2 inGlyphSize;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inOffset;
layout(location = 4) in float inScale;

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
    // Use inGlyphSize to clip unnecessary quad space thus improving performance
    vec2 unit_pos = vec2(gl_VertexIndex >> 1, gl_VertexIndex & 1) * inGlyphSize;

    vs_out.texCoord = vec2(unit_pos.x, 1.0 - unit_pos.y);
    vs_out.glyphIndex = inGlyphIndex;
    vs_out.color = inColor;
    vs_out.pxRange = pxRange;

    // Negate inOffset.y because text starts at top-left origin, but we need bottom-left
    vec2 pos2d = unit_pos * inScale + vec2(inOffset.x, -inOffset.y);
    vec4 worldPos = params.transform * vec4(pos2d, 0.0, 1.0);

    gl_Position = projView * worldPos;
}