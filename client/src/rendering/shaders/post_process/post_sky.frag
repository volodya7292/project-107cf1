#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../../../engine/shaders/sky.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outAlbedo;

layout(binding = 0) uniform sampler2D previousComposition;
layout(binding = 1) uniform sampler2D backPositionImage;
layout(binding = 2) uniform sampler2D backDepthImage;

layout(binding = 3, scalar) uniform FrameInfoBlock {
    FrameInfo info;
};

void main() {
    vec4 backAlbedo = texture(previousComposition, inUV);
    float backDepth = texture(backDepthImage, inUV).r;

    vec3 mainColor = vec3(0);
    if (backDepth < 0.0001) {
        vec3 sun_dir = info.main_light_dir.xyz;
        vec3 skyCol = calculateSky(inUV, info.frame_size, info.camera.pos.xyz, info.camera.dir.xyz, info.camera.fovy, info.camera.view, sun_dir);

        float dist = 10000;
        float fog = 1 - exp(-pow(-1./50. * dist, 10));
        fog *= 0.5;

        mainColor = mix(skyCol, vec3(0.7), fog);
    }

    mainColor.rgb = mix(mainColor, backAlbedo.rgb, backAlbedo.a);

    outAlbedo = vec4(mainColor, 1);
}
