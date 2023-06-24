#version 450
#extension GL_GOOGLE_include_directive : require

layout(early_fragment_tests) in;

#define FN_TEXTURE_ATLAS
#define ENGINE_PIXEL_SHADER
#include "../../../engine/shaders/common.glsl"

const vec3 ambient_light = vec3(0.6);

layout(location = 0) in Output {
    vec2 tex_uv;
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    flat uint material_id;
    float ao;
    vec3 light;
    vec3 sky_light;
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

void sample_material(uint id, vec2 coord, out SampledMaterial sampled_mat) {
    Material mat = materials[id];

    sampled_mat.diffuse = vec4(0);
    sampled_mat.specular = vec4(0);
    sampled_mat.emission = mat.emission.rgb;
    sampled_mat.normal = vec3(0);

    // Diffuse
    if (mat.diffuse_tex_id == -1) {
        sampled_mat.diffuse = mat.diffuse;
    } else {
        sampled_mat.diffuse += textureAtlas(albedoAtlas, info.tex_atlas_info.x, coord, mat.diffuse_tex_id);
    }

    // Specular
    if (mat.specular_tex_id == -1) {
        sampled_mat.specular = mat.specular;
    } else {
        sampled_mat.specular += textureAtlas(specularAtlas, info.tex_atlas_info.x, coord, mat.specular_tex_id);
    }

    // Normal
    if (mat.normal_tex_id == -1) {
        sampled_mat.normal = vs_in.surface_normal;
    } else {
        // Tangent space normal maps
        vec3 normal = textureAtlas(normalAtlas, info.tex_atlas_info.x, coord, mat.normal_tex_id).xyz * 2.0 - 1.0;

        // // Swizzle world normals to match tangent space and apply RNM blend
        // normal = rnm_blend_unpacked(vec3(normal_coord[i], abs(vs_in.surface_normal[i])), normal);
        // // Reapply sign to Z
        // normal.z *= sign(vs_in.surface_normal[i]);

        // sampled_mat.normal += normal.xyz * triplan_weight[i];
        // if (i == 2)
        //     sampled_mat.normal = normalize(sampled_mat.normal + vs_in.surface_normal);
    }
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

    // SampledMaterial samples[3][1];
    // vec3 diffuse_s[3];

    // for (uint i = 0; i < 3; i++) {
    //     diffuse_s[i] = vec3(0);
    //     for (uint j = 0; j < 1; j++) {
    //         sample_material(vs_in.material_ids[i][j], samples[i][j]);
    //         diffuse_s[i] += samples[i][j].diffuse.rgb / 1.0;
    //     }
    // }

    // vec3 diffuse = (diffuse_s[0] * vs_in.barycentrics[0]
    //     + diffuse_s[1] * vs_in.barycentrics[1]
    //     + diffuse_s[2] * vs_in.barycentrics[2]).rgb;

    SampledMaterial mat;

    uint material_id;
    sample_material(vs_in.material_id, vs_in.tex_uv, mat);

//    vec3 combined_light = min(vec3(1.0), ambient_light + vs_in.light);
    vec3 combined_light = max(vs_in.sky_light, vs_in.light);
//    vec3 combined_light = min(vec3(1.0), vs_in.light);
//    vec3 combined_light = min(vec3(1.0), ambient_light + vs_in.light);

//    vec3 diffuse = mat.diffuse.rgb * combined_light * max(0.75, vs_in.ao);
//    vec3 diffuse = mat.diffuse.rgb * combined_light * max(0.75, vs_in.ao);

    float aoMin = 0.75;
    float ao = 1 - (1 - vs_in.ao) * aoMin;
    vec3 diffuse = mat.diffuse.rgb * ao * combined_light;
    vec3 emission = vec3(0);//mat.diffuse.rgb * combined_light;

    writeOutput(vs_in.world_pos, vec4(diffuse.rgb, mat.diffuse.a), vec4(0.0), emission, vs_in.surface_normal);
}
