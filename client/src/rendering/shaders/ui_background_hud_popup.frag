#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "ui.glsl"
#include "../../../engine/shaders/object3d.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO, scalar) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    float opacity;
    float corner_radius;
    vec4 color;
};

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    float sd = roundedRectSDF(normScreenCoord, clip_rect, corner_radius, info.frame_size / info.scale_factor);
    if (sd > 1.0) {
        discard;
    }

    vec2 normCoord = vs_in.texCoord;
    float density = smoothstep(0, 0.5, 1.0 - sd);

    vec4 finalColor = vec4(color.rgb, color.a * density * opacity);

    writeOutput(finalColor);
}
