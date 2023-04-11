#version 450

layout(binding = 0) uniform sampler2D srcTexture;
layout(binding = 1) uniform sampler2D bloomTexture;

//layout(push_constant) uniform PushConstants {
//
//};

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec3 outColor;


void main() {
    vec3 src = texture(srcTexture, texCoord).rgb;
    vec3 bloom = texture(bloomTexture, texCoord).rgb;

    outColor = mix(src, bloom, 0.05);
}
