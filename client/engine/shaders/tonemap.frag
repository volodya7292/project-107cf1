#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(binding = 0) uniform sampler2D mainTexture;
layout(binding = 1) uniform sampler2D overlayTexture;
layout(binding = 2) uniform sampler2D bloomTexture;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec3 outColor;

vec3 tonemap_exp(vec3 v) {
    float l_old = luminance(v);
    float l_new = 1 - exp(-2.0f * l_old);
    return change_luminance(v, l_new);
}

vec3 normalize_balance(vec3 v) {
    float max_comp = max(v.r, max(v.g, v.b));
    float max_norm = max(1.0, max_comp);
    return v / max_norm;
}

void main() {
    vec3 mainColor = texture(mainTexture, texCoord).rgb;
    vec4 overlayColor = texture(overlayTexture, texCoord);
    vec3 bloom = texture(bloomTexture, texCoord).rgb;

    vec3 main_tonemapped = tonemap_exp(mainColor);
    vec3 overlay_normalized = normalize_balance(overlayColor.rgb);

    outColor = mix(main_tonemapped, overlay_normalized, overlayColor.a);
    outColor.rgb = mix(outColor.rgb, bloom, 0.02);
}
