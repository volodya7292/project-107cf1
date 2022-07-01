#version 450

layout(early_fragment_tests) in;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = 0, binding = 0) uniform sampler2DArray msdfArray;

layout(location = 0) in Input {
    vec2 inUV;
    uint inGlyphIndex;
    vec4 inColor;
};

layout(push_constant) uniform PushConstants {
    float pxRange;
    mat3 transform;
    mat4 projView;
} params;

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

void main() {
    vec3 msd = texture(msdfArray, vec3(inUV, inGlyphIndex)).rgb;
    float sd = median(msd.r, msd.g, msd.b);
    float screenPxDistance = params.pxRange * (sd - 0.5);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    outColor = vec4(inColor.rgb, inColor.a * opacity);
}