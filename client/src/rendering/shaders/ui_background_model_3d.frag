#version 450
#extension GL_GOOGLE_include_directive : require

#define FN_TEXTURE_ATLAS
#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "ui.glsl"
#include "../../../engine/shaders/object3d.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO, scalar) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    float opacity;
    float corner_radius;
    uint materialId;
};

layout(location = 0) in Input {
    vec2 texCoord;
    vec3 surfaceNormal;
    vec3 localPos;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect, corner_radius, info.frame_size / info.scale_factor)) {
        discard;
    }

    vec2 triplan_coord = vs_in.surfaceNormal.z * vs_in.localPos.xy
                    + vs_in.surfaceNormal.y * vs_in.localPos.xz
                    + vs_in.surfaceNormal.x * vs_in.localPos.yz;

    Material mat = materials[materialId];
    vec4 diffuse = textureAtlas(albedoAtlas, info.tex_atlas_info.x, triplan_coord, mat.diffuse_tex_id); 

    // vec3 lighting = vec3(
    //     0.23 * dot(vs_in.surfaceNormal, vec3(1, 0, 0)),
    //     0.33 * dot(vs_in.surfaceNormal, vec3(0, 1, 0)),
    //     0.43 * dot(vs_in.surfaceNormal, vec3(0, 0, 1))
    // );
    // lighting = abs(lighting);
    // float intensity = (lighting.x + lighting.y + lighting.z);
    // float intensity = dot(info.camera.dir.xyz, vs_in.surfaceNormal);
    float intensity = max(0, dot(-info.main_light_dir.xyz, vs_in.surfaceNormal));
    intensity = mix(0.6, 1.0, intensity);

    vec4 outColor = vec4(diffuse.rgb * intensity, opacity);

    writeOutput(outColor);
}
