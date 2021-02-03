#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#define FN_TEXTURE_ATLAS
#include "common.glsl"

#define NODE_TYPE_INTERNAL 0
#define NODE_TYPE_LEAF 1

struct EncodedNode {
    uint8_t ty;
    uint children[8];
    uint16_t material[8];
};

layout(early_fragment_tests) in;

layout(location = 0) out vec4 outDiffuse;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = 0, binding = 0) uniform per_frame_data {
    PerFrameInfo info;
};

layout(set = 0, binding = 1) uniform sampler2D albedoAtlas;
layout(set = 0, binding = 2) uniform sampler2D specularAtlas;
layout(set = 0, binding = 3) uniform sampler2D normalAtlas;

layout(set = 0, binding = 4, std430) readonly buffer Materials {
    Material materials[];
};

layout(set = 1, binding = 1, std430) readonly buffer PerSectorData {
    uint size;
    EncodedNode nodes[];
} sector;

layout(location = 0) in Output {
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    vec2 tex_coord;
} vs_in;

vec2 triplan_coord[3];
vec2 normal_coord[3];
vec3 triplan_weight;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

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
        if (mat.diffuse_tex_id == -1) {
            sampled_mat.diffuse = mat.diffuse;
        } else {
            sampled_mat.diffuse += textureAtlas(albedoAtlas, info.tex_atlas_info.x, triplan_coord[i], mat.diffuse_tex_id) * triplan_weight[i];
        }

        // Specular
        if (mat.specular_tex_id == -1) {
            sampled_mat.specular = mat.specular;
        } else {
            sampled_mat.specular += textureAtlas(specularAtlas, info.tex_atlas_info.x, triplan_coord[i], mat.specular_tex_id) * triplan_weight[i];
        }

        // Normal
        if (mat.normal_tex_id == -1) {
            sampled_mat.normal = vs_in.surface_normal;
        } else {
            // Tangent space normal maps
            vec3 normal = textureAtlas(normalAtlas, info.tex_atlas_info.x, triplan_coord[i], mat.normal_tex_id).xyz * 2.0 - 1.0;

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

void sample_node(vec3 local_pos, out uint mats[8], out vec3 norm_pos_in_node) {
    uint curr_node_id = 0;
    uint curr_size = sector.size + 2;
    vec3 curr_pos = local_pos / curr_size;

    while (curr_size >= 1) {
        EncodedNode curr_node = sector.nodes[curr_node_id];

        if (curr_node.ty == NODE_TYPE_INTERNAL) {
            curr_size >>= 1;

            uvec3 child_pos = uvec3(greaterThanEqual(curr_pos, 0.5.xxx));
            uint child_index = (child_pos.x << 2) + (child_pos.y << 1) + child_pos.z;

            curr_pos = (curr_pos - 0.5.xxx * child_pos) * 2.0;
            curr_node_id = curr_node.children[child_index];
        } else {
            for (uint i = 0; i < 8; i++)
                mats[i] = curr_node.material[i];
            norm_pos_in_node = curr_pos;
            break;
        }
    }
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
    vec3 color = clamp((vs_in.world_pos.xyz + 128) / 256.0, 0, 1);
    color = vec3(pow(color.r * color.g * color.b, 1. / 3.), 1, 1);
    color = hsv2rgb(color);
    color = (vs_in.surface_normal + 1.0) / 2.0;

    if (color.r != color.r)
        color = vec3(1.0);    

    // Calculate triplanar texture mapping
    // -------------------------------------------------------
    triplan_weight = abs(vs_in.surface_normal);
    triplan_weight = max((triplan_weight - 0.2) * 7, 0);
    triplan_weight /= (triplan_weight.x + triplan_weight.y + triplan_weight.z);

    triplan_coord[0] = vs_in.world_pos.yz;
    triplan_coord[1] = vs_in.world_pos.zx;
    triplan_coord[2] = vs_in.world_pos.xy;

    normal_coord[0] = vs_in.surface_normal.yz;
    normal_coord[1] = vs_in.surface_normal.zx;
    normal_coord[2] = vs_in.surface_normal.xy;
    // -------------------------------------------------------



    uint mats[8];
    vec3 norm_pos_in_node;
    sample_node(vs_in.local_pos, mats, norm_pos_in_node);

    float coeffs[8];
    calc_trilinear_unit_coeffs(norm_pos_in_node, coeffs);

    vec4 diffuse = vec4(0);

    for (uint i = 0; i < 8; i++) {
        SampledMaterial sampled_material;
        sample_material(mats[i], sampled_material);

        diffuse += sampled_material.diffuse * coeffs[i];
        // diffuse += vec4(color, 1) * coeffs[i];
    }
    

    //outDiffuse = textureAtlas(albedoAtlas, info.tex_atlas_info.x, vs_out.tex_coord, 0);
    // outDiffuse = vec4(color, 1);
    outDiffuse = vec4(diffuse.xyz, 1);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}