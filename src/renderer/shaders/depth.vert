#version 450
#extension GL_GOOGLE_include_directive : require
#include "GBufferCommon.glsl"

// IA input
layout(location = 0) in vec3 inPosition;

// Payload
layout(binding = 0) uniform camera_data {
    Camera camera;
};
layout(binding = 1) uniform per_object_data {
    mat4 model;
};

// layout(location = 0) out float Output {
//     float flogz;
// } vs_out;

void main() {
    gl_Position = camera.proj_view * (model * vec4(inPosition, 1));
    //gl_Position.z = log2(max(1e-6, 1.0 + gl_Position.w)) * Fcoef - 1.0;
}
