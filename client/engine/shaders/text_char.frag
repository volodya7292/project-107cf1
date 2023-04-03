#version 450
#extension GL_GOOGLE_include_directive : require

layout(early_fragment_tests) in;

#define ENGINE_PIXEL_SHADER
#include "../../engine/shaders/common.glsl"

layout(set = SET_CUSTOM_PER_FRAME, binding = 1) uniform sampler2DArray msdfArray;

layout(location = 0) in Input {
    vec2 texCoord;
    flat uint glyphIndex;
    vec4 color;
    float pxRange;
} vs_in;

struct Rect {
    vec2 min;
    vec2 max;
};

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
};

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

    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (any(lessThan(normScreenCoord, clip_rect.min)) || any(greaterThan(normScreenCoord, clip_rect.max))) {
        discard;
    }

    writeOutputAlbedo(vec4(vs_in.color.rgb, vs_in.color.a * opacity));
}
