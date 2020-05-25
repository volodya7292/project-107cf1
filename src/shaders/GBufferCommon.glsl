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

vec4 infi_clip(vec4 clip_coord, float dist) {
    // clip_coord.xy /= clip_coord.w;
    clip_coord.y = -clip_coord.y;
    // clip_coord.z = -dist / 65536.0f;
    // clip_coord.w = 1;
    return clip_coord;
}