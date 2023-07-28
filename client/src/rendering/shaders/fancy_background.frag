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
    vec4 color;
};

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect)) {
        discard;
    }

    vec2 pixelSize = 1.0 / vec2(info.frame_size);
    vec2 diff = fwidth(vs_in.texCoord) * info.scale_factor * 2;

    if (vs_in.texCoord.x > diff.x && (1 - vs_in.texCoord.y) > diff.y) {
        discard;
    }

    vec4 finalColor = color;
    finalColor.a *= opacity;

    writeOutput(finalColor);
}
