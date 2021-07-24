#version 450

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D albedo;
 
void main() {
    outColor = texture(albedo, inUV);//  vec4(0, 0.5, 0.1, 1.0);
}