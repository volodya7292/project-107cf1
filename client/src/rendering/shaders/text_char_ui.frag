#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../../engine/shaders/text_char_frag_template.glsl"

struct Rect {
    vec2 min;
    vec2 max;
};

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

    writeOutputAlbedo(vec4(color, vs_in.color.a * opacity));
}
