#include "common.glsl"

#ifdef ENGINE_PIXEL_SHADER

#ifndef RENDER_GBUFFER_TRANSPARENCY
#define RENDER_GBUFFER_TRANSPARENCY 0
#endif

#ifndef RENDER_DEPTH_ONLY
#define RENDER_DEPTH_ONLY 0
#endif

#ifndef RENDER_CLOSEST_DEPTHS
#define RENDER_CLOSEST_DEPTHS 0
#endif

#ifdef ENGINE_PIXEL_SHADER_UI
layout(location = 0) out vec4 outAlbedo;
#else
#if RENDER_DEPTH_ONLY == 0
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outAlbedo;
layout(location = 2) out vec4 outSpecular;
layout(location = 3) out vec4 outEmission;
layout(location = 4) out vec4 outNormal;
#endif
#endif

#if RENDER_CLOSEST_DEPTHS == 1
layout (input_attachment_index = 0, set = SET_GENERAL_PER_FRAME, binding = BINDING_SOLID_DEPTHS) uniform subpassInput inputSolidDepth;
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
#if RENDER_DEPTH_ONLY == 0
void writeOutputAlbedo(vec4 albedo) {
    if (!bool(RENDER_GBUFFER_TRANSPARENCY) || albedo.a < ALPHA_BIAS) {
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
#else
void writeOutputAlbedo(vec4 albedo) {}
#endif

#if RENDER_CLOSEST_DEPTHS == 1
void writeClosestDepths() {
    // Prevent Z-fighting. Note: using reversed-Z
    {
        float solidDepth = subpassLoad(inputSolidDepth).r;
        solidDepth = linearize_depth(solidDepth, info.camera.z_near);
        float fragDepth = linearize_depth(gl_FragCoord.z, info.camera.z_near);

        // Do not render transparency where depth is uncertain
        if (fragDepth < solidDepth - 0.001 * solidDepth) {
            return;
        }
    }

    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint currDepth = floatBitsToUint(gl_FragCoord.z);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint sliceSize = info.frame_size.x * info.frame_size.y;

    // Early transparency depth test (check farthest layer)
    uint lastLayerIdx = OIT_N_CLOSEST_LAYERS - 1;
    uint lastDepth = depthsArray[coordIdx + lastLayerIdx * sliceSize];
    if (currDepth < lastDepth) {
        return;
    }

    // Find OIT_N_CLOSEST_LAYERS closest depths and append new depths if possible
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        uint prev = atomicMax(depthsArray[coordIdx + i * sliceSize], currDepth);
        if (prev == FARTHEST_DEPTH_UINT || prev == currDepth) {
            break;
        }
        currDepth = min(prev, currDepth);
    }
}
#else
void writeClosestDepths() {}
#endif

void writeOutput(vec3 position, vec4 albedo, vec4 specular, vec3 emission, vec3 normal) {
#if RENDER_DEPTH_ONLY == 0
    outPosition.rgb = position;
    outSpecular = specular;
    outEmission.rgb = emission;
    outNormal.xy = normalToSphericalAngles(normal);
    writeOutputAlbedo(albedo);
#endif

    writeClosestDepths();
}
#endif
#endif // ENGINE_PIXEL_SHADER

#ifdef ENGINE_VERTEX_SHADER

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
    // if (PASS_TYPE == PASS_TYPE_DEPTH_MAIN_SHADOW_MAP) {
    //     gl_Position = shadowClipPosMapping(position);
    // } else {
    gl_Position = position;
    // }
}
#endif // ENGINE_VERTEX_SHADER