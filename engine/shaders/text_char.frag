#version 450

layout(early_fragment_tests) in;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = 0, binding = 1) uniform sampler2DArray msdfArray;

layout(location = 0) in Input {
    vec2 texCoord;
    uint glyphIndex;
    vec4 color;
    float pxRange;
} vs_in;

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

float screenPxRange() {
    vec2 unitRange = vs_in.pxRange.xx / textureSize(msdfArray, 0).xy;
    vec2 screenTexSize = vec2(1.0) / fwidth(vs_in.texCoord);
    return max(0.5 * dot(unitRange, screenTexSize), 1.0);
}

void main() {
    vec3 msd = texture(msdfArray, vec3(vs_in.texCoord, vs_in.glyphIndex)).rgb;
    float sd = median(msd.r, msd.g, msd.b);
    float screenPxDistance = screenPxRange() * (sd - 0.5);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    outColor = vec4(vs_in.color.rgb, vs_in.color.a * opacity);
}