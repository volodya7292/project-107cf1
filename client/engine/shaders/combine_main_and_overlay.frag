#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outCombinedColor;

layout(binding = 0) uniform sampler2D gMainAlbedo;
layout(binding = 1) uniform sampler2D gOverlayAlbedo;

void main() {
    vec4 mainColor = texture(gMainAlbedo, inUV);
    vec4 overlayColor = texture(gOverlayAlbedo, inUV);

    vec3 combinedColor = mix(mainColor.rgb, overlayColor.rgb, overlayColor.a);

    outCombinedColor = vec4(combinedColor, 1);
}
