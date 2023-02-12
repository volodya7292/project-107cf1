#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#include "common.glsl"

void main() {
    outAlbedo = vec4(0.3, 0.4, 0.6, 0.7);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}
