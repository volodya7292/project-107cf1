#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#include "GBufferCommon.glsl"

#define DENSITY_BUFFER_SIZE 64u // SxSxS

// IA input
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in uint inDensityMatIndex;
// //[FIRST 3 byte: cluster-relative x/y/z cube index, LAST byte: density index]

// Output
layout(location = 0) out Output {
    vec3 world_pos;
    vec3 surface_normal;
    uvec4 cube_mat_ids;
} vs_out;

// Payload
layout(binding = 0) uniform camera_data {
    Camera camera;
};
layout(binding = 1) uniform per_object_data {
    mat4 model;
};
layout(std430, binding = 2) readonly buffer buffers {
    // 8 16bit material id values (for each cube index)
    uvec4 density_mats[];
} density_buffers[];

layout(push_constant) uniform push_constants {
    uint buffers_index;
};

void main() {
    vs_out.world_pos      = (model * vec4(inPosition, 1)).xyz;
    vs_out.surface_normal = (model * vec4(inNormal, 1)).xyz;
    vs_out.cube_mat_ids = density_buffers[buffers_index].density_mats[inDensityMatIndex];

    gl_Position   = camera.proj_view * vec4(vs_out.world_pos, 1);
}
