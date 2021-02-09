#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in gl_PerVertex {
    vec4 gl_Position;
} gl_in[];

layout(location = 0) in Input {
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    uint16_t material_ids[8];
} vs_in[];

layout(location = 0) out Output {
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    vec3 barycentrics;
    uint16_t material_ids[3][8];
} gs_out;

void main() {
    gs_out.material_ids[0] = vs_in[0].material_ids;
    gs_out.material_ids[1] = vs_in[1].material_ids;
    gs_out.material_ids[2] = vs_in[2].material_ids;

    gl_Position = gl_in[0].gl_Position;
    gs_out.local_pos = vs_in[0].local_pos;
    gs_out.world_pos = vs_in[0].world_pos;
    gs_out.surface_normal = vs_in[0].surface_normal;
    gs_out.barycentrics = vec3(1, 0, 0); 
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    gs_out.local_pos = vs_in[1].local_pos;
    gs_out.world_pos = vs_in[1].world_pos;
    gs_out.surface_normal = vs_in[1].surface_normal;
    gs_out.barycentrics = vec3(0, 1, 0); 
    EmitVertex();

    gl_Position = gl_in[2].gl_Position;
    gs_out.local_pos = vs_in[2].local_pos;
    gs_out.world_pos = vs_in[2].world_pos;
    gs_out.surface_normal = vs_in[2].surface_normal;
    gs_out.barycentrics = vec3(0, 0, 1); 
    EmitVertex();

    EndPrimitive();
}