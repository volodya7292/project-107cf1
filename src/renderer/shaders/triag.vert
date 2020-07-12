#version 450
#extension GL_GOOGLE_include_directive : require
#include "GBufferCommon.glsl"

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform camera_data {
    Camera camera;
};
layout(set = 1, binding = 0) uniform per_object_data {
    mat4 model;
};

void main() {
    gl_Position = camera.proj_view * (model * vec4(inPosition, 1));
}