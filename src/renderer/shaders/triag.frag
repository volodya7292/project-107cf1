#version 450
layout(early_fragment_tests) in;

layout(location = 0) out vec4 outDiffuse;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = 0, binding = 1) uniform sampler2D albedoAtlas;

vec2 calc_atlas_tex_coords(uint index) {
    return vec2(0);
}

vec4 sample_atlas(vec2 offset) {
    return vec4(1);
}

void main() {
    outDiffuse = texture(albedoAtlas, vec2(0.0, 0.0));
    //outDiffuse = vec4(0.2, 0.5, 0.9, 1);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}