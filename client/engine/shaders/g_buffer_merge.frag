#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D gPosition;
layout(binding = 1) uniform sampler2D gAlbedo;
layout(binding = 2) uniform sampler2D gSpecular;
layout(binding = 3) uniform sampler2D gEmissive;
layout(binding = 4) uniform sampler2D gNormal;
layout(binding = 5) uniform sampler2D gDepth;

layout(binding = 6, scalar) uniform FrameData {
    FrameInfo info;
};

layout(binding = 7, std430) coherent buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(binding = 8, rgba8) uniform image2DArray translucencyColorsArray;

layout(binding = 9) uniform sampler2DArray mainShadowMap;

const vec2 meanSampleJitter[] = {
    vec2(0, 0),
    vec2(0, -1),
    vec2(0, 1),
    vec2(-1, 0),
    vec2(1, 0),
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, -1),
    vec2(1, 1),
};

float calc_shadow(vec3 worldPos) {
    vec4 worldPosLightClipSpace = info.main_light.proj_view * vec4(worldPos, 1);
    worldPosLightClipSpace.y = -worldPosLightClipSpace.y;

    vec3 worldPosLightNDC = worldPosLightClipSpace.xyz / worldPosLightClipSpace.w;
    vec2 worldPosLightNorm = worldPosLightNDC.xy * 0.5 + 0.5;
    float depthAtWorldPos = worldPosLightNDC.z;

//    float meanDepth = 0.0;
//    float minDepth = depthAtWorldPos;
//    float maxDepth = depthAtWorldPos;
    vec2 texelSize = 1.0 / textureSize(mainShadowMap, 0).xy;

    float bias = 0.0001;
    float shadowAccum = 0.0;

    for (uint i = 0; i < 4; i++) {
        vec2 relUV = worldPosLightNorm.xy + texelSize * meanSampleJitter[i];
        float shadowMapDepth = texture(mainShadowMap, vec3(relUV, 0.0)).r;

        shadowAccum += 1.0 - float((depthAtWorldPos - bias) > shadowMapDepth);

//        shadowMapDepth *= ;

//        vec4 g = textureGather(mainShadowMap, relUV, 0);

//        bvec4 d = greaterThan(g,) g >= 0.0;

//        meanDepth += shadowMapDepth;
//        minDepth = min(minDepth, shadowMapDepth);
//        maxDepth = max(maxDepth, shadowMapDepth);
    }
    shadowAccum /= 9.0;

//    if (maxDepth - minDepth < 0.00001) {
//
//    }

//    float param1 = max(depthAtWorldPos - maxDepth, 0.00001) / max(maxDepth - minDepth, 0.00001);

//    float shadow = (meanDepth - minDepth) / max(maxDepth - minDepth, 0.00001);
//    float shadow = clamp(param1, 0.0, 1.0);

//    float depthFromLight = texture(mainShadowMap, worldPosLightNorm.xy).r;

//    float bias = 0.0001;
//    float bias = max(0.05 * (1.0 - dot(surfaceNormal, -info.main_light.dir)), 0.005);
//    float shadow = (depthAtWorldPos - bias) > meanDepth ? 0.0 : 1.0;

    return 1.0;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint depthSliceSize = info.frame_size.x * info.frame_size.y;

    vec4 currColor = vec4(0);

    // Collect translucency
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        if (depthsArray[coordIdx + i * depthSliceSize] == 0xFFFFFFFFu) {
            // The following layers do not contain any colors, stop the loop
            break;
        } else {
            vec4 nextColor = imageLoad(translucencyColorsArray, ivec3(coord, i));
            // Note: reverse blending
            currColor.rgb = mix(nextColor.rgb, currColor.rgb, currColor.a);
            currColor.a = currColor.a + (1 - currColor.a) * nextColor.a;
        }
    }

    vec3 worldPos = texture(gPosition, inUV).rgb;
    float depth = texture(gDepth, inUV).r;

    // Blend with solid colors
    vec4 solidColor = texture(gAlbedo, inUV);
    currColor = mix(solidColor, currColor, currColor.a);

    vec4 emission = texture(gEmissive, inUV);
    currColor.rgb += emission.rgb;

    float shadow = calc_shadow(worldPos);

    outColor = vec4(currColor.rgb * shadow, 1);
}
