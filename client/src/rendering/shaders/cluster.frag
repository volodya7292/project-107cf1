#version 450
#extension GL_GOOGLE_include_directive : require

layout(early_fragment_tests) in;

#define FN_TEXTURE_ATLAS
#define ENGINE_PIXEL_SHADER
#include "../../../engine/shaders/object3d.glsl"

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
vec2 triplan_coord2;
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

vec3 hash33( vec3 p )
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
			  dot(p,vec3(269.5,183.3,246.1)),
			  dot(p,vec3(113.5,271.9,124.6)));

	return fract(sin(p)*43758.5453123);
}

struct InterpNodes2
{
    vec2 seeds;
    vec2 weights;
};
InterpNodes2 GetNoiseInterpNodes(float smoothNoise)
{
    vec2 globalPhases = vec2(smoothNoise * 0.5) + vec2(0.5, 0.0);
    vec2 phases = fract(globalPhases);
    vec2 seeds = floor(globalPhases) * 2.0 + vec2(0.0, 1.0);
    vec2 weights = min(phases, vec2(1.0f) - phases) * 2.0;
    return InterpNodes2(seeds, weights);
}
vec4 GetTextureSample(in sampler2D samp, uint tile_width, vec2 pos, uint texId, float freq, float seed)
{
    vec3 hash = hash33(vec3(seed, 0.0, 0.0));
    float ang = hash.x * 2.0 * M_PI;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));

    vec2 uv = rotation * pos * freq + hash.yz;
    return textureAtlas(samp, tile_width, uv, texId);
}

// 2D Random
float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))
                 * 43758.5453123);
}

float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

vec4 textureNoTile(in sampler2D samp, uint tile_width, vec2 uv, uint texId ) {
    float smoothNoise = noise(uv);

    uint layersCount = 5;
    InterpNodes2 interpNodes = GetNoiseInterpNodes(smoothNoise * layersCount);
    float moment2 = 0.0;
    vec4 col = vec4(0);
    for(int i = 0; i < 2; i++)
    {
        float weight = interpNodes.weights[i];
        moment2 += weight * weight;
        col += GetTextureSample(samp, tile_width, uv, texId, 1, interpNodes.seeds[i]) * weight;
    }
    return col;
}

void sample_material(uint id, vec2 coord, out SampledMaterial sampled_mat) {
    Material mat = materials[id];

    sampled_mat.diffuse = vec4(0);
    sampled_mat.specular = vec4(0);
    sampled_mat.emission = mat.emission.rgb;
    sampled_mat.normal = vec3(0);

    // Diffuse
//    if (mat.diffuse_tex_id == -1) {
//        sampled_mat.diffuse = mat.diffuse;
//    } else {
//        sampled_mat.diffuse += textureAtlas(albedoAtlas, info.tex_atlas_info.x, coord, mat.diffuse_tex_id);
//    }
    // for (uint i = 0; i < 3; i++) {
    //     if (mat.diffuse_tex_id == -1) {
    //         sampled_mat.diffuse = mat.diffuse;
    //     } else {
    //         sampled_mat.diffuse += textureNoTile(albedoAtlas, info.tex_atlas_info.x, triplan_coord[i], mat.diffuse_tex_id) * triplan_weight[i];
    //     }
    // }
    if (mat.diffuse_tex_id == -1) {
        sampled_mat.diffuse = mat.diffuse;
    } else {
        sampled_mat.diffuse += textureNoTile(albedoAtlas, info.tex_atlas_info.x, triplan_coord2, mat.diffuse_tex_id);
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

    triplan_coord2 = vs_in.surface_normal.z * vs_in.world_pos.xy
                    + vs_in.surface_normal.y * vs_in.world_pos.xz
                    + vs_in.surface_normal.x * vs_in.world_pos.yz;


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

    float aoMin = 0.4;
    float ao = 1 - (1 - vs_in.ao) * aoMin;
    vec3 diffuse = mat.diffuse.rgb * ao * combined_light;
    vec3 emission = vec3(0);//mat.diffuse.rgb * combined_light;

    writeOutput(vs_in.world_pos, vec4(diffuse.rgb, mat.diffuse.a), vec4(0.0), emission, vs_in.surface_normal);
}
