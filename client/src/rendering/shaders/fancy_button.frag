#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#define ENGINE_PIXEL_SHADER_UI
#include "../../../engine/shaders/common.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    vec4 color;
};

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    writeOutput(color);
}
