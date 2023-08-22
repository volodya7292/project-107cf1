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
    // ------------
    float health_factor;
    float satiety_factor;
};

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

const vec3 HEALTH_COLOR = SRGB2LIN(vec3(0.65, 0.12, 0.12));
const vec3 SATIETY_COLOR = SRGB2LIN(vec3(0.65, 0.65, 0.12));

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, corner_radius, info.frame_size * info.scale_factor)) {
        discard;
    }

    vec2 normCoord = vec2(vs_in.texCoord.x, 1 - vs_in.texCoord.y);
    float dist = length(normCoord * 2 - 1);
    float dDist = fwidth(dist);

    float innerCutoutVis = extractIsosurface(dist, dDist, 0.9);
    float outerCutoutVis = 1.0 - extractIsosurface(dist, dDist, 1.0);
    float combinedVis = innerCutoutVis * outerCutoutVis;

    float health_level = 0.5 * health_factor + 0.01 * sin(normCoord.x * 20.0 + info.time);
    float satiety_level = health_level + 0.5 * satiety_factor;

    vec3 color = vec3(0.2);
    if (normCoord.y <= health_level) {
        color = HEALTH_COLOR;
    } else if (normCoord.y <= satiety_level) {
        color = SATIETY_COLOR;
    }

    vec4 finalColor = vec4(color, combinedVis * opacity);

    writeOutput(finalColor);
}
