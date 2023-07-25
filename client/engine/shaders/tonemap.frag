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
