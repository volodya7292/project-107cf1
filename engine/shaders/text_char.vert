#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(location = 0) in uint inGlyphIndex;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inTransformCol0;
layout(location = 3) in vec3 inTransformCol1;
layout(location = 4) in vec3 inTransformCol2;

layout(set = 0, binding = 0) uniform PerFrameData {
    PerFrameInfo info;
};

layout(location = 0) out Output {
    vec2 texCoord;
    uint glyphIndex;
    vec4 color;
} vs_out;

void main() {
    vs_out.texCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vs_out.glyphIndex = inGlyphIndex;
    vs_out.color = inColor;

    mat3 transform = mat3(inTransformCol0, inTransformCol1, inTransformCol2);
    vec3 worldPos = transform * vec3(vs_out.texCoord * 2.0f - 1.0f, 0.0f);

    gl_Position = info.camera.proj_view * vec4(worldPos, 1.0);
//    gl_Position.y = -gl_Position.y;
}