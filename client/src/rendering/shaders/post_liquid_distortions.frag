#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../../engine/shaders/common.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outAlbedo;

layout(binding = 0) uniform sampler2D backPosition;
layout(binding = 1) uniform sampler2D backAlbedoImage;
layout(binding = 2) uniform sampler2D backDepth;

layout(binding = 3, scalar) uniform FrameInfoBlock {
    FrameInfo info;
};

layout(binding = 4, scalar) uniform CustomFrameInfoBlock {
    bool enabled;
};

const float FREQ = 10.0;
const float INTENSITY = 0.003;
const float VISIBILITY = 50;
const vec3 LIQUID_COLOR = vec3(0, 0, 1);

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    vec2 clipScreenCoord = normScreenCoord * 2.0 - 1.0;
    vec4 mainColor = vec4(0);

    if (enabled) {
        float dist = max(abs(clipScreenCoord.x), abs(clipScreenCoord.y));
        float smoothFactor = 1.0 - pow(dist, 4);

        vec2 jitter = vec2(
            sin(clipScreenCoord.y * FREQ + info.time) * INTENSITY,
            sin(clipScreenCoord.x * FREQ + info.time + 10.0) * INTENSITY
        );
        vec2 distortedUV = inUV + jitter * smoothFactor;
        vec3 backPosition = texture(backAlbedoImage, distortedUV).xyz;
        vec4 backAlbedo = texture(backAlbedoImage, distortedUV);
        float backDepth = texture(backDepth, distortedUV).r;

        mainColor = backAlbedo;
        mainColor.rgb = mix(mainColor.rgb, LIQUID_COLOR, 0.2);

        float fog = pow(1.0 - backDepth, VISIBILITY);
        mainColor.rgb = mix(mainColor.rgb, LIQUID_COLOR, fog);
    } else {
        mainColor = texture(backAlbedoImage, inUV);
    }

    outAlbedo = mainColor;
}
