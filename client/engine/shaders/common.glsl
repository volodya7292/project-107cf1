#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : require

#include "engine_ids.h"

#define ALPHA_BIAS 4.0 / 255.0
#define OIT_N_CLOSEST_LAYERS 4
#define THREAD_GROUP_WIDTH 8
#define THREAD_GROUP_HEIGHT 8
#define THREAD_GROUP_1D_WIDTH (THREAD_GROUP_WIDTH * THREAD_GROUP_HEIGHT)

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
    mat4 proj_view;
    float z_near;
    float fovy;
};

struct FrameInfo {
    Camera camera;
    uvec4 tex_atlas_info; // .x: tile size in pixels
    uvec2 frame_size;
};


#ifdef ENGINE_PIXEL_SHADER
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outSpecular;
layout(location = 2) out vec4 outEmission;
layout(location = 3) out vec4 outNormal;

layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_FRAME_INFO, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_MATERIAL_BUFFER, scalar) readonly buffer Materials {
    Material materials[];
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_ALBEDO_ATLAS) uniform sampler2D albedoAtlas;
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_SPECULAR_ATLAS) uniform sampler2D specularAtlas;
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_NORMAL_ATLAS) uniform sampler2D normalAtlas;

layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_TRANSPARENCY_DEPTHS, std430) coherent buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_TRANSPARENCY_COLORS, rgba8) uniform image2DArray translucencyColorsArray;

layout(push_constant) uniform PushConstants {
    uint isTranslucentPass;
};

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

void writeOutputAlbedo(vec4 albedo) {
    if (isTranslucentPass == 0 || albedo.a < ALPHA_BIAS) {
        // Do not render translucency, this is solid colors pass
        outAlbedo = albedo;
        return;
    }

    uint currDepth = floatBitsToUint(gl_FragCoord.z);
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint sliceSize = info.frame_size.x * info.frame_size.y;
    uint lastLayerIdx = OIT_N_CLOSEST_LAYERS - 1;

    if (currDepth > depthsArray[coordIdx + lastLayerIdx * sliceSize]) {
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
        if (currDepth <= depth) {
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

#endif // ENGINE_PIXEL_SHADER


// R2 low discrepancy sequence
// ----------------------------------------------------------------------------------------
float r2_seq_1d(uint n) {
    return fract(0.5 + 0.618033988749 * float(n));
}

vec2 r2_seq_2d(uint n) {
    return fract(0.5 + vec2(0.754877666246, 0.569840290998) * float(n));
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