#version 450

layout(binding = 0) uniform sampler2D mainTexture;
layout(binding = 1) uniform sampler2D overlayTexture;
layout(binding = 2) uniform sampler2D bloomTexture;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec3 outColor;

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / max(l_in, 0.001));
}

vec3 tonemap_exp(vec3 v) {
    float l_old = luminance(v);
    float l_new = 1 - exp(-2.0f * l_old);
    return change_luminance(v, l_new);
}

void main() {
    vec3 mainColor = texture(mainTexture, texCoord).rgb;
    vec4 overlayColor = texture(overlayTexture, texCoord);
    vec3 bloom = texture(bloomTexture, texCoord).rgb;

    vec3 main_tonemapped = tonemap_exp(mainColor);

    outColor = mix(main_tonemapped, overlayColor.rgb, overlayColor.a);
    outColor.rgb = mix(outColor.rgb, bloom, 0.02);
}
