#version 450
#extension GL_GOOGLE_include_directive : require

#define FN_TEXTURE_ATLAS
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

layout(location = 0) in Output {
    vec3 world_pos;
    vec2 tex_coord;
} vs_out;

void main() {


    //outDiffuse = textureAtlas(albedoAtlas, info.tex_atlas_info.x, vs_out.tex_coord, 0);
    outDiffuse = vec4(vs_out.world_pos.xyz / 65.0, 1);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}