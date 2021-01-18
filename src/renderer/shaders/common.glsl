#define THREAD_GROUP_WIDTH 8
#define THREAD_GROUP_HEIGHT 8
#define THREAD_GROUP_1D_WIDTH (THREAD_GROUP_WIDTH * THREAD_GROUP_HEIGHT)

#define M_PI 3.1415927f
#define SQRT_2 1.4142136f

#define FLT_MAX 3.402823466e+38f

struct Material {
    uvec4 texture_ids;// .x - diffuse, .y - specular, .z - normal
// if texture_id == -1 then the parameters below are used
    vec4 diffuse;// .a - opacity/translucency
    vec4 specular;// .a - roughness
    vec4 emission;
};

struct SampledMaterial {
    vec4 diffuse;// .a - opacity/translucency
    vec4 specular;// .a - roughtness
    vec3 emission;
    vec3 normal;
};

struct Vertex {
    vec3 pos;
    vec3 tex_coord;// .z - material id
    vec3 normal;
    vec3 tangent;
};

struct Camera {
    vec4 pos;
    vec4 dir;
    mat4 proj;
    mat4 view;
    mat4 proj_view;
    vec4 info;// .x - FovY
};

struct PerFrameInfo {
    Camera camera;
    uvec4 tex_atlas_info; // .x: tile size in pixels
};

#ifdef FN_TEXTURE_ATLAS
/// tile_width: single tile width in pixels
/// tex_coord: regular texture coordinates
vec4 textureAtlas(in sampler2D atlas, uint tile_width, vec2 tex_coord, uint tile_index) {
    ivec2 atlas_size = textureSize(atlas, 0);
    uint size_in_tiles = atlas_size.x / tile_width;

    // Calculate LOD
    vec2 dx = dFdx(tex_coord * tile_width);
    vec2 dy = dFdy(tex_coord * tile_width);
    float d = max(dot(dx, dx), dot(dy, dy));
    float lod = 0.5 * log2(d);
    
    // Calculate texture coordinates
    float tile_width_norm = 1.0 / size_in_tiles;
    float pixel_size = 1.0 / tile_width;

    vec2 tile_offset = vec2(
        (tile_index % size_in_tiles) * tile_width_norm,
        (tile_index / size_in_tiles) * tile_width_norm
    );
    float pixel_offset = pixel_size * (0.5 * pow(2.0, lod));

    tex_coord -= floor(tex_coord); // repeat pattern
    tex_coord = clamp(tex_coord, pixel_offset, 1.0 - pixel_offset); // remove bleeding
    tex_coord = tile_offset + tex_coord * tile_width_norm; // adjust to proper tile

    return textureLod(atlas, tex_coord, lod);
}
#endif


// R2 low discrepancy sequence
// ----------------------------------------------------------------------------------------
float r2_noise_1d(uint n) {
    return fract(0.5 + 0.618033988749 * float(n));
}

vec2 r2_noise_2d(uint n) {
    return fract(0.5 + vec2(0.754877666246, 0.569840290998) * float(n));
}

vec3 r2_noise_3d(uint n) {
    return fract(0.5 + vec3(0.819172513396, 0.671043606703, 0.549700477901) * float(n));
}
// ----------------------------------------------------------------------------------------

