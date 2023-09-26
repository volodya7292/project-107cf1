#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../../engine/shaders/common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
};

layout(location = 0) out Output {
    vec2 texCoord;
    vec3 surfaceNormal;
} vs_out;

void main() {
    vec3 localPos = 0.5 * (inPosition + 1.0);
    vec4 worldPos = model * vec4(localPos, 1.0);

    // vs_out.texCoord = vec2(pos2d.x, 1.0 - pos2d.y);
    vs_out.surfaceNormal = inNormal;

    gl_Position = info.camera.proj_view * worldPos;
}
