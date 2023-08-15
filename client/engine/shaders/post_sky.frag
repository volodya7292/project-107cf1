#version 450
#extension GL_GOOGLE_include_directive : require
#include "sky.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outAlbedo;

layout(binding = 0) uniform sampler2D backAlbedoImage;
layout(binding = 1) uniform sampler2D backDepth;

layout(binding = 2, scalar) uniform FrameInfoBlock {
    FrameInfo info;
};

void main() {
    vec4 backAlbedo = texture(backAlbedoImage, inUV);
    float backDepth = texture(backDepth, inUV).r;

    vec3 mainColor = vec3(0);
    if (backDepth < 0.0001) {
        vec3 sun_dir = info.main_light_dir.xyz;
        vec3 skyCol = calculateSky(inUV, info.frame_size, info.camera.pos.xyz, info.camera.dir.xyz, info.camera.fovy, info.camera.view, sun_dir);
        mainColor = skyCol;
    }

    mainColor.rgb = mix(mainColor, backAlbedo.rgb, backAlbedo.a);

    outAlbedo = vec4(mainColor, 1);
}
