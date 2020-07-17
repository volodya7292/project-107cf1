#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(early_fragment_tests) in;

layout(location = 0) out vec4 outDiffuse;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = 0, binding = 0) uniform per_frame_data {
    PerFrameInfo info;
};

layout(set = 0, binding = 1) uniform sampler2D albedoAtlas;

layout(location = 0) in vec2 tex_coord;

vec2 calc_atlas_coord(vec2 tex_coord, uint tex_index) {
    float tile_size = 1.0 / info.tex_atlas_info.x;
    float pixel_size = 1.0 / info.tex_atlas_info.y;

    vec2 offset = vec2(
        (tex_index % info.tex_atlas_info.x) * tile_size,
        (tex_index / info.tex_atlas_info.x) * tile_size
    );

    tex_coord -= floor(tex_coord); // repeat pattern
    //tex_coord += pixel_size * 0.5; // half pixel correction
    tex_coord = clamp(tex_coord, pixel_size * 2, 1.0 - pixel_size * 2);
    tex_coord = offset + tex_coord * tile_size;
    return tex_coord;
}

void main() {
    // TODO: custom lod calculation (to fix bleading texture edges)
    // https://community.khronos.org/t/texture-lod-calculation-useful-for-atlasing/61475
    
    outDiffuse = textureLod(albedoAtlas, calc_atlas_coord(tex_coord, 0), 0.0);
    //outDiffuse = vec4(0.2, 0.5, 0.9, 1);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}