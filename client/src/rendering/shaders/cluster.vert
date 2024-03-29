#version 450
#extension GL_GOOGLE_include_directive : require
#define ENGINE_VERTEX_SHADER
#include "../../../engine/shaders/object3d.glsl"

#define CLUSTER_SIZE 24

layout(location = 0) in uvec4 inPack1;
layout(location = 1) in uint inPack2;

layout(set = SET_GENERAL_PER_FRAME, binding = BINDING_FRAME_INFO, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
};

layout(location = 0) out Output {
    vec2 tex_uv;
    vec3 local_pos;
    vec3 world_pos;
    vec3 surface_normal;
    uint material_id;
    float ao;
    vec3 light;
    vec3 sky_light;
} vs_out;

vec3 decodeLight(uint rawLight) {
    return vec3(
        float((rawLight >> 10) & 0x1Fu),
        float((rawLight >> 5) & 0x1Fu),
        float(rawLight & 0x1Fu)
    ) / 32.0;
}

void main() {
    vec3 inPosition = vec3(uvec3((inPack1[0] >> 16) & 0xffffu, (inPack1[0]) & 0xffffu, (inPack1[1] >> 16) & 0xffffu));
    inPosition = inPosition / 65535.0 * CLUSTER_SIZE;
    vec3 inNormal = vec3(uvec3((inPack1[1] >> 8) & 0xffu, inPack1[1] & 0xffu, (inPack1[2] >> 24) & 0xffu));
    inNormal = inNormal / 255.0 * 2.0 - 1.0;
    vec2 inTexUV = vec2(uvec2((inPack1[3] >> 16) & 0xffffu, inPack1[3] & 0xffffu));
    inTexUV = inTexUV / 65535.0 * 64.0;
    float inAO = float((inPack1[2] >> 16) & 0xffu) / 255.0;
    uint inMaterialId = inPack1[2] & 0xffffu;

    uint inLighting = inPack2 & 0xffffu;
    uint inSkyLighting = (inPack2 >> 16) & 0xffffu;

    vec3 light = decodeLight(inLighting);
    vec3 skyLight = decodeLight(inSkyLighting);

    vec4 world_pos = (model * vec4(inPosition.xyz, 1));

    vs_out.tex_uv = inTexUV;
    vs_out.local_pos = inPosition.xyz;
    vs_out.world_pos = world_pos.xyz;
    vs_out.surface_normal = inNormal;
    vs_out.material_id = inMaterialId;
    vs_out.ao = inAO;
    vs_out.light = light;
    vs_out.sky_light = skyLight;

    writeOutput(info.camera.proj_view * world_pos);
}
