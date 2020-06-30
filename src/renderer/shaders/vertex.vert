#version 450
#extension GL_GOOGLE_include_directive : require
#include "GBufferCommon.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(binding = 0) uniform uniform_camera {
    Camera camera;
};

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragTexColor;
layout(location = 2) out vec4 fragCoord;
layout(location = 3) out vec3 fragNormal;

void main() {
    vec4 world_pos = vec4(inPosition, 1);

    fragTexCoord = world_pos.xz / 16.0f;
    fragTexColor = vec4(world_pos.xyz / 16.0f, 1);
    fragCoord = world_pos;
    fragNormal = inNormal;
    gl_Position = infi_clip(camera.proj_view * world_pos, distance(world_pos.xyz, camera.pos.xyz));
}