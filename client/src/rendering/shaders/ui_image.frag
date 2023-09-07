#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "ui.glsl"
#include "../../../engine/shaders/object3d.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO, scalar) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    vec2 img_offset;
    vec2 img_scale;
    float opacity;
};

layout(set = SET_PER_OBJECT, binding = CUSTOM_OBJ_BINDING_START_ID) uniform sampler2D sourceImage;

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, 0.0, info.frame_size / info.scale_factor)) {
        discard;
    }
    // Remove repetitions
    if (any(lessThan(vs_in.texCoord, vec2(0))) || any(greaterThan(vs_in.texCoord, vec2(1)))) {
        discard;
    }

    vec4 col = texture(sourceImage, vs_in.texCoord);
    col.a *= opacity;

    writeOutput(col);
}
