#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER_UI
#include "ui.glsl"
#include "../../../engine/shaders/text_char_frag_template.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    float opacity;
    float innerShadowIntensity;
};

// `value` and `midpoint` are in range [0, 1]; `strength`: [0, +inf]
float localize(float value, float midpoint, float strength) {
    float hp;

    if (value > midpoint) {
        hp = 1.0 / (1.0 + 2.0 * (value - midpoint) * strength);
    } else {
        hp = 1.0 + 2.0 * (midpoint - value) * strength;
    }

    return pow(value, hp);
}

void main() {
    float aa_alpha, sd;
    calculateCharShading(0.5, aa_alpha, sd);

    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, 0.0, info.frame_size / info.scale_factor)) {
        discard;
    }

    vec4 color = vs_in.color;
//    float innerShadow = mix(1.0, sd, innerShadowIntensity);
//    color.rgb *= 1.0 - localize(sd, 0.100, 10.0);

    if (aa_alpha < ALPHA_BIAS) {
        discard;
    }

    if (aa_alpha == 1.0) {
        writeOutput(vec4(color.rgb, color.a * opacity));
    } else {
        // Clamp the RGB values so that antialiasing alpha is in SDR
        writeOutput(vec4(min(color.rgb, 1.0), color.a * aa_alpha * opacity));
    }
}
