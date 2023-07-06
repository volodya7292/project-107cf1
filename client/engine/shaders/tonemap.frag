#version 450

layout(binding = 0) uniform sampler2D srcTexture;
layout(binding = 1) uniform sampler2D bloomTexture;

//layout(push_constant) uniform PushConstants {
//
//};

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec3 outColor;

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

vec3 tonemap_exp(vec3 v) {
    float l_old = luminance(v);
    float l_new = 1 - exp(-2 * l_old);
    return change_luminance(v, l_new);
}

void main() {
    vec3 src = texture(srcTexture, texCoord).rgb;
    vec3 bloom = texture(bloomTexture, texCoord).rgb;

    vec3 src_tonemapped = tonemap_exp(src);

    outColor = mix(src_tonemapped, bloom, 0.02);
}
