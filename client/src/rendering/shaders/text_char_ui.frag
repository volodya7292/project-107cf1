#version 450
#extension GL_GOOGLE_include_directive : require

#include "ui.glsl"
#include "../../../engine/shaders/text_char_frag_template.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    float innerShadowIntensity;
};

void main() {
    float opacity, sd;
    calculateCharShading(opacity, sd);

    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (any(lessThan(normScreenCoord, clip_rect.min)) || any(greaterThan(normScreenCoord, clip_rect.max))) {
        discard;
    }

    vec3 color = vs_in.color.rgb;
    float innerShadow = mix(1.0, sd, innerShadowIntensity);

    if (sd > 0.01) {
       color *= innerShadow;
    }

    if (opacity < ALPHA_BIAS) {
        discard;
    }

    if (opacity == 1.0) {
        outEmission = vec4(10) * opacity;
    } else {
        outEmission = vec4(vec3(opacity), 1);
    }

//    writeOutputAlbedo(vec4(color, vs_in.color.a * opacity));
}
