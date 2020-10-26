#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#include "common.glsl"

// Vertex input
layout(location = 0) in VertexInput {
    vec3 world_pos;
    vec3 surface_normal;
    flat uvec4 cube_mat_ids;
} vs_in;

// Output
layout(location = 0) out vec4 outDiffuse;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

// Payload
layout(binding = 3) uniform sampler2D images[];
layout(std430, binding = 4) readonly buffer Materials {
    Material materials[];
};

layout(push_constant) uniform push_constants {
    uint govno;
    uint srach_with_govno;
};

vec2 triplan_coord[3];
vec2 normal_coord[3];
vec3 triplan_weight;

// Reoriented Normal Mapping (Unity 3D)
// http://discourse.selfshadow.com/t/blending-in-detail/21/18
vec3 rnm_blend_unpacked(vec3 n1, vec3 n2) {
    n1 += vec3(0, 0, 1);
    n2 *= vec3(-1, -1, 1);
    return n1 * dot(n1, n2) / n1.z - n2;
}

void sample_material(uint id, out SampledMaterial sampled_mat) {
    Material mat = materials[id];

    sampled_mat.diffuse = vec4(0);
    sampled_mat.specular = vec4(0);
    sampled_mat.emission = mat.emission.rgb;
    sampled_mat.normal = vec3(0);

    for (uint i = 0; i < 3; i++) {
        // Diffuse
        if (mat.texture_ids[0] == -1) {
            sampled_mat.diffuse = mat.diffuse;
        } else {
            sampled_mat.diffuse += texture(images[mat.texture_ids[0]], triplan_coord[i]) * triplan_weight[i];
        }

        // Specular
        if (mat.texture_ids[1] == -1) {
            sampled_mat.specular = mat.specular;
        } else {
            sampled_mat.specular += texture(images[mat.texture_ids[1]], triplan_coord[i]) * triplan_weight[i];
        }

        // Normal
        if (mat.texture_ids[2] == -1) {
            sampled_mat.normal = vs_in.surface_normal;
        } else {
            // Tangent space normal maps
            vec3 normal = texture(images[mat.texture_ids[2]], triplan_coord[i]).xyz * 2.0f - 1.0f;

            // Swizzle world normals to match tangent space and apply RNM blend
            normal = rnm_blend_unpacked(vec3(normal_coord[i], abs(vs_in.surface_normal[i])), normal);
            // Reapply sign to Z
            normal.z *= sign(vs_in.surface_normal[i]);

            sampled_mat.normal += normal.xyz * triplan_weight[i];
            if (i == 2)
                sampled_mat.normal = normalize(sampled_mat.normal + vs_in.surface_normal);
        }
    }
}

// TODO: change to trilinear interpolation
void cube_interpolate(vec3 p, out float v[9]) {
    v[0] = 1.0f - distance(p, vec3(0, 0, 0));
    v[1] = 1.0f - distance(p, vec3(1, 0, 0));
    v[2] = 1.0f - distance(p, vec3(0, 1, 0));
    v[3] = 1.0f - distance(p, vec3(1, 1, 0));
    v[4] = 1.0f - distance(p, vec3(0, 0, 1));
    v[5] = 1.0f - distance(p, vec3(1, 0, 1));
    v[6] = 1.0f - distance(p, vec3(0, 1, 1));
    v[7] = 1.0f - distance(p, vec3(1, 1, 1));
    v[8] = 1.0f / (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]);
}

void main() {
    // Calculate factors of 8 cube corner textures
    float inter_values[9];
    // cube_interpolate(vs_in.cube_interpol_pos, inter_values);
    cube_interpolate(vs_in.world_pos - floor(vs_in.world_pos), inter_values);

    // Calculate triplanar texture mapping
    triplan_weight = abs(vs_in.surface_normal);
    triplan_weight = max((triplan_weight - 0.2) * 7, 0);
    triplan_weight /= (triplan_weight.x + triplan_weight.y + triplan_weight.z);

    triplan_coord[0] = vs_in.world_pos.yz;
    triplan_coord[1] = vs_in.world_pos.zx;
    triplan_coord[2] = vs_in.world_pos.xy;

    normal_coord[0] = vs_in.surface_normal.yz;
    normal_coord[1] = vs_in.surface_normal.zx;
    normal_coord[2] = vs_in.surface_normal.xy;

    outDiffuse = outSpecular = outEmission = outNormal = vec4(0);

    SampledMaterial temp_mat;
    for (uint i = 0; i < 4; i++) {
        sample_material(vs_in.cube_mat_ids[i] & 0xffu, temp_mat); // low half 16 bit
        outDiffuse += temp_mat.diffuse * inter_values[i * 2];
        outSpecular += temp_mat.specular * inter_values[i * 2];
        outEmission.xyz += temp_mat.emission * inter_values[i * 2];
        outNormal.xyz += temp_mat.normal * inter_values[i * 2];

        sample_material(vs_in.cube_mat_ids[i] >> 8u, temp_mat); // high half 16 bit
        outDiffuse += temp_mat.diffuse * inter_values[i * 2 + 1];
        outSpecular += temp_mat.specular * inter_values[i * 2 + 1];
        outEmission.xyz += temp_mat.emission * inter_values[i * 2 + 1];
        outNormal.xyz += temp_mat.normal * inter_values[i * 2 + 1];
    }

    // Normalize cube interpolation
    outDiffuse *= inter_values[8];
    outSpecular *= inter_values[8];
    outEmission *= inter_values[8];
    outNormal.xyz = normalize(outNormal.xyz * inter_values[8]);
}
