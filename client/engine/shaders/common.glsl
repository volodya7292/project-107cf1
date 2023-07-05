#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : require

#include "engine_ids.h"

#define ALPHA_BIAS 4.0 / 255.0

#define M_PI 3.1415927f
#define SQRT_2 1.4142136f
#define FLT_MAX 3.402823466e+38f

struct Material {
    uint diffuse_tex_id;
    uint specular_tex_id;
    uint normal_tex_id;
    // if texture_id == -1 then the parameters below are used
    vec4 diffuse; // .a - opacity/translucency
    vec4 specular; // .a - roughness
    vec4 emission;
};

struct SampledMaterial {
    vec4 diffuse; // .a - opacity/translucency
    vec4 specular; // .a - roughtness
    vec3 emission;
    vec3 normal;
};

struct Vertex {
    vec3 pos;
    vec3 tex_coord; // .z - material id
    vec3 normal;
    vec3 tangent;
};

struct Camera {
    vec4 pos;
    vec4 dir;
    mat4 proj;
    mat4 view;
    mat4 viewInverse;
    mat4 proj_view;
    float z_near;
    float fovy;
};

struct LightInfo {
    mat4 proj_view;
    vec4 dir;
};

struct FrameInfo {
    Camera camera;
    vec4 main_light_dir;
    uvec4 tex_atlas_info; // .x: tile size in pixels
    uvec2 frame_size;
    uvec2 surface_size;
};

vec2 normalToSphericalAngles(vec3 normal) {
    normal = normalize(normal);

    float theta = acos(normal.z);
    float phi = atan(normal.y, normal.x);
    return vec2(theta, phi);
}

vec3 sphericalAnglesToNormal(vec2 angles) {
    return vec3(
        sin(angles.x) * cos(angles.y),
        sin(angles.x) * sin(angles.y),
        cos(angles.x)
    );
}

vec4 shadowClipPosMapping(vec4 pos) {
    float maxDepth = 256;
    vec2 worldPos = pos.xy * maxDepth;

    float stepX = 64;
    float maxX = stepX * log2(1 + maxDepth);

    pos.xy = sign(pos.xy) * (stepX * log2(1 + abs(worldPos)) / maxX);

    return pos;
}

#ifdef ENGINE_PIXEL_SHADER
layout(constant_id = CONST_ID_PASS_TYPE) const uint PASS_TYPE = 0;

#ifdef ENGINE_PIXEL_SHADER_UI
layout(location = 0) out vec4 outAlbedo;
#else
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outAlbedo;
layout(location = 2) out vec4 outSpecular;
layout(location = 3) out vec4 outEmission;
layout(location = 4) out vec4 outNormal;
#endif

layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_FRAME_INFO, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_MATERIAL_BUFFER, scalar) readonly buffer Materials {
    Material materials[];
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_ALBEDO_ATLAS) uniform sampler2D albedoAtlas;
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_SPECULAR_ATLAS) uniform sampler2D specularAtlas;
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_NORMAL_ATLAS) uniform sampler2D normalAtlas;

#ifndef ENGINE_PIXEL_SHADER_UI
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_TRANSPARENCY_DEPTHS, std430) coherent buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_TRANSPARENCY_COLORS, rgba8) uniform image2DArray translucencyColorsArray;
#endif

/// tile_width: with of single tile in pixels
/// tex_coord: regular texture coordinates
vec4 textureAtlas(in sampler2D atlas, uint tile_width, vec2 tex_coord, uint tile_index) {
    ivec2 atlas_size = textureSize(atlas, 0);
    float mip_levels = max(log2(float(tile_width)), 3) - 2; // account for BC block size (4x4)
    uint size_in_tiles = atlas_size.x / tile_width;
    float tile_width_norm = 1.0 / size_in_tiles;
    float pixel_size = 1.0 / tile_width;
    vec2 tile_offset = vec2(tile_index % size_in_tiles, tile_index / size_in_tiles);

    // Calculate LOD
    vec2 tex_coord_pixels = tex_coord * tile_width;
    vec2 dx = dFdx(tex_coord_pixels);
    vec2 dy = dFdy(tex_coord_pixels);
    float d = max(dot(dx, dx), dot(dy, dy));
    float lod = clamp(0.5 * log2(d), 0, mip_levels - 1);

    // Calculate texture coordinates
    float pixel_offset = pixel_size * pow(2.0, lod);

    tex_coord = fract(tex_coord); // repeat pattern
    tex_coord = clamp(tex_coord, pixel_offset, 1.0 - pixel_offset); // remove bleeding
    tex_coord = (tile_offset + tex_coord) * tile_width_norm;  // adjust to proper tile position

    return textureLod(atlas, tex_coord, lod);
}



#ifdef ENGINE_PIXEL_SHADER_UI
void writeOutput(vec4 albedo) {
    outAlbedo = albedo;
}
#else
void writeOutputAlbedo(vec4 albedo) {
    if (PASS_TYPE != PASS_TYPE_G_BUFFER_TRANSLUCENCY || albedo.a < ALPHA_BIAS) {
        // Do not render translucency, this is solid colors pass
        outAlbedo = albedo;
        return;
    }

    uint currDepth = floatBitsToUint(gl_FragCoord.z);
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint sliceSize = info.frame_size.x * info.frame_size.y;
    uint lastLayerIdx = OIT_N_CLOSEST_LAYERS - 1;

    // note: z is reversed
    if (currDepth < depthsArray[coordIdx + lastLayerIdx * sliceSize]) {
        // The fragment falls behind closest depths => perform tail-blending
        outAlbedo = albedo;
        return;
    }

    uint start = 0;
    uint end = lastLayerIdx;

    // Binary search: find suitable layer index based on closest depth
    while (start < end) {
        uint mid = (start + end) / 2;
        uint depth = depthsArray[coordIdx + mid * sliceSize];
        if (currDepth >= depth) {
            end = mid;
        } else {
            start = mid + 1;
        }
    }

    // Insert albedo at corresponding index
    // Note: this causes false positive error from vulkan synchronizaton validation
    imageStore(translucencyColorsArray, ivec3(coord, start), albedo);
    outAlbedo = vec4(0);
}

void writeOutput(vec3 position, vec4 albedo, vec4 specular, vec3 emission, vec3 normal) {
    outPosition.rgb = position;
    outSpecular = specular;
    outEmission.rgb = emission;
    outNormal.xy = normalToSphericalAngles(normal);

    writeOutputAlbedo(albedo);
}
#endif
#endif // ENGINE_PIXEL_SHADER

#ifdef ENGINE_VERTEX_SHADER
layout(constant_id = CONST_ID_PASS_TYPE) const uint PASS_TYPE = 0;

//layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_FRAME_INFO, scalar) uniform FrameData {
//    FrameInfo info;
//};
//layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
//    mat4 model;
//};
//
//struct VertexOuts {
//    vec2 tex_uv;
//    vec3 local_pos;
//    vec3 world_pos;
//    vec3 world_pos_from_main_light;
//    vec3 surface_normal;
//    uint material_id;
//};
//
//
//


void writeOutput(vec4 position) {
    if (PASS_TYPE == PASS_TYPE_DEPTH_MAIN_SHADOW_MAP) {
        gl_Position = shadowClipPosMapping(position);
    } else {
        gl_Position = position;
    }
}
#endif // ENGINE_VERTEX_SHADER


// R2 low discrepancy sequence
// ----------------------------------------------------------------------------------------
float r2_seq_1d(uint n) {
    return fract(0.5 + 0.618033988749 * float(n));
}

vec2 r2_seq_2d(uint n) {
    return fract(0.5 + vec2(0.754877666246, 0.569840290998) * float(n % 1024));
}

vec3 r2_seq_3d(uint n) {
    return fract(0.5 + vec3(0.819172513396, 0.671043606703, 0.549700477901) * float(n));
}
// ----------------------------------------------------------------------------------------

void calc_trilinear_unit_coeffs(vec3 p, out float v[8]) {
    vec3 np = 1.0 - p;
    vec4 xy = vec4(np.x, p.x, np.x, p.x) * vec4(np.y, np.y, p.y, p.y);
    vec4 xyz0 = xy * np.z;
    vec4 xyz1 = xy * p.z;

    // v[0] = xyz0[0];
    // v[1] = xyz0[1];
    // v[2] = xyz0[2];
    // v[3] = xyz0[3];
    // v[4] = xyz1[0];
    // v[5] = xyz1[1];
    // v[6] = xyz1[2];
    // v[7] = xyz1[3];

    v[0] = xyz0[0];
    v[1] = xyz1[0];
    v[2] = xyz0[2];
    v[3] = xyz1[2];
    v[4] = xyz0[1];
    v[5] = xyz1[1];
    v[6] = xyz0[3];
    v[7] = xyz1[3];
}

// For projection with infinite far-plane and depth in range (0;1)
float linearize_depth(float d, float z_near) {
    return z_near / (1.0 - d);
}
