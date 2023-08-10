#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "ui.glsl"
#include "../../../engine/shaders/common.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO, scalar) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    float opacity;
    float corner_radius;
    vec4 filter_color;
    float pain_factor;
};

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, corner_radius, info.frame_size * info.scale_factor)) {
        discard;
    }

    vec2 pixelSize = 1.0 / vec2(info.frame_size);
    float painDistCutoff = 0.5 * sqrt(2.0f) - 1.0;

    float painDist = distance(vec2(0.5), normScreenCoord);
    painDist -= painDistCutoff;
    float pxPainFactor = clamp(painDist, 0, 1);
    pxPainFactor = pow(pxPainFactor, 3);

    vec4 finalColor = filter_color;
    finalColor.a *= opacity;

    finalColor = mix(finalColor, vec4(0.4, 0.01, 0.01, 1.0), pxPainFactor * pain_factor);

    writeOutput(finalColor);
}
