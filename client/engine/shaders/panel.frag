#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "common.glsl"

void main() {
    writeOutput(vec4(0.3, 0.4, 0.6, 0.7));
}
