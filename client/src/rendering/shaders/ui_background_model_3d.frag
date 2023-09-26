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
    vec3 surfaceNormal;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, corner_radius, info.frame_size / info.scale_factor)) {
        discard;
    }

    // vec3 lighting = vec3(
    //     0.23 * dot(vs_in.surfaceNormal, vec3(1, 0, 0)),
    //     0.33 * dot(vs_in.surfaceNormal, vec3(0, 1, 0)),
    //     0.43 * dot(vs_in.surfaceNormal, vec3(0, 0, 1))
    // );
    // lighting = abs(lighting);
    // float intensity = (lighting.x + lighting.y + lighting.z);
    float intensity = dot(info.camera.dir.xyz, vs_in.surfaceNormal);

    // lighting = mix(0.6, 1.0, lighting); 

    // vec2 pixelSize = 1.0 / vec2(info.frame_size);
    // vec2 diff = fwidth(vs_in.texCoord) * info.scale_factor * 2;

    // if (vs_in.texCoord.x > diff.x && (1 - vs_in.texCoord.y) > diff.y) {
    //     discard;
    // }

    // vec4 finalColor = color;
    // finalColor.a *= opacity;

    writeOutput(vec4(vec3(intensity), opacity));
}
