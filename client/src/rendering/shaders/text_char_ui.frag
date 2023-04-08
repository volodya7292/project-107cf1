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
};

void main() {
    float opacity = calculateOpacity();

    if (PASS_TYPE == PASS_TYPE_G_BUFFER_OVERLAY) {
        vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
        if (any(lessThan(normScreenCoord, clip_rect.min)) || any(greaterThan(normScreenCoord, clip_rect.max))) {
            discard;
        }
    }

    writeOutputAlbedo(vec4(vs_in.color.rgb, vs_in.color.a * opacity));
}
