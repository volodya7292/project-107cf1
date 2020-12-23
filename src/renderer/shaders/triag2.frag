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
    vec3 normal;
    vec2 tex_coord;
} vs_out;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void calc_trilinear_unit_coeffs(vec3 p, out float v[8]) {
    vec3 np = 1.0 - p;
    vec4 xy = vec4(np.x, p.x, np.x, p.x) * vec4(np.y, np.y, p.y, p.y);
    vec4 xyz0 = xy * np.z;
    vec4 xyz1 = xy * p.z;

    v[0] = xyz0[0];
    v[1] = xyz0[1];
    v[2] = xyz0[2];
    v[3] = xyz0[3];
    v[4] = xyz1[0];
    v[5] = xyz1[1];
    v[6] = xyz1[2];
    v[7] = xyz1[3];
}

void main() {
    vec3 color = clamp((vs_out.world_pos.xyz + 128) / 256.0, 0, 1);
    color = vec3(pow(color.r * color.g * color.b, 1. / 3.), 1, 1);
    color = hsv2rgb(color);
    color = (vs_out.normal + 1.0) / 2.0;

    if (color.r != color.r)
        color = vec3(1.0);
 

    //outDiffuse = textureAtlas(albedoAtlas, info.tex_atlas_info.x, vs_out.tex_coord, 0);
    outDiffuse = vec4(color, 1);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}