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
    float scale_factor;
    float time;
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

// For projection with infinite far-plane and depth in range (0;1)
float linearize_depth(float d, float z_near) {
    return z_near / (1.0 - d);
}

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



// Converts srgb color (usually used by devs) to linear (for correct calculations/interpolations)
#define SRGB2LIN(rgb) pow(rgb, vec3(2.2))

// Accepts signed distance field value `v` and its pixel-delta `dV` (such as fwidth).
// Result: `v` < isoValue => 0, `v` >= isoValue => 1.
// Makes the transition between 0 and 1 smooth at the pixel level (antialised).
float extractIsosurface(float v, float dV, float isoValue) {
    return smoothstep(isoValue - dV, isoValue + dV, v);
}

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / max(l_in, 0.001));
}

vec3 adjust_luminance(vec3 c_in, float l_delta) {
    float l_in = luminance(c_in);
    return change_luminance(c_in, clamp(l_in + l_delta, 0, 1));
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - clamp(cosTheta, 0, 1), 5.0);
}