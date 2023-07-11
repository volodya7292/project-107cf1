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

void main() {
    float aa_alpha, sd;
    calculateCharShading(aa_alpha, sd);

    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect)) {
        discard;
    }

    vec4 color = vs_in.color;
    float innerShadow = mix(1.0, sd, innerShadowIntensity);

    if (sd > 0.01) {
       color *= innerShadow;
    }

    if (aa_alpha < ALPHA_BIAS) {
        discard;
    }

    if (aa_alpha == 1.0) {
        writeOutput(vec4(color.rgb, color.a * opacity));
    } else {
        writeOutput(vec4(vec3(1), aa_alpha * opacity));
    }

//    writeOutputAlbedo(vec4(color, vs_in.color.a * opacity));
}
