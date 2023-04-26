#version 450
#extension GL_GOOGLE_include_directive : require
#include "sky.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D gPosition;
layout(binding = 1) uniform sampler2D gAlbedo;
layout(binding = 2) uniform sampler2D gSpecular;
layout(binding = 3) uniform sampler2D gEmissive;
layout(binding = 4) uniform sampler2D gNormal;
layout(binding = 5) uniform sampler2D gDepth;

layout(binding = 6, scalar) uniform FrameInfoBlock {
    FrameInfo info;
};

layout(binding = 7, std430) readonly buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(binding = 8, rgba8) uniform image2DArray translucencyColorsArray;

layout(binding = 9) uniform sampler2D mainShadowMap;

layout(binding = 10, scalar) uniform MainShadowInfoBlock {
    mat4 lightView;
    mat4 lightProjView;
    vec4 lightDir;
} mainShadowInfo;


float calc_shadow(vec3 worldPos, vec3 normal) {

    vec4 worldPosLightClipSpace = mainShadowInfo.lightProjView * vec4(worldPos, 1);
    worldPosLightClipSpace = shadowClipPosMapping(worldPosLightClipSpace);
    worldPosLightClipSpace.y = -worldPosLightClipSpace.y;

    vec3 worldPosLightNDC = worldPosLightClipSpace.xyz / worldPosLightClipSpace.w;
    vec2 worldPosLightNorm = worldPosLightNDC.xy * 0.5 + 0.5;
    float depthAtWorldPos = worldPosLightNDC.z;

    float shadowFactor = 0;
    float pfcRange = 4;
    uint samples = 8;

    uint texId = floatBitsToUint(worldPos.x) ^ floatBitsToUint(worldPos.y) ^ floatBitsToUint(worldPos.z);
    ivec2 texSize = textureSize(mainShadowMap, 0).xy;
    vec2 texelSize = 1.0 / texSize;

    vec4 normalOffset = mainShadowInfo.lightView * vec4(normal, 0);
    normalOffset.y = -normalOffset.y;

    for (int i = 0; i < samples; i++) {
    	vec2 jitter = r2_seq_2d(texId + i) * 2.0 - 1.0;
    	jitter += normalOffset.xy;
    	jitter *= pfcRange;

    	vec2 offset = texelSize * jitter;
        float shadowMapDepth = texture(mainShadowMap, worldPosLightNorm.xy + offset).r;

        float bias = 0.0001;// max(0.5 * (1.0 - dot(normal, -mainShadowInfo.lightDir.xyz)), 0.001);

        float shadow = float((depthAtWorldPos + bias) > shadowMapDepth);
        shadowFactor += shadow;
    }

    shadowFactor /= samples;

    return shadowFactor;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint depthSliceSize = info.frame_size.x * info.frame_size.y;

    vec4 transpColor = vec4(0);

    // Collect translucency
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        if (depthsArray[coordIdx + i * depthSliceSize] == 0xFFFFFFFFu) {
            // The following layers do not contain any colors, stop the loop
            break;
        } else {
            vec4 nextColor = imageLoad(translucencyColorsArray, ivec3(coord, i));
            // Note: reverse blending
            transpColor.rgb = mix(nextColor.rgb, transpColor.rgb, transpColor.a);
            transpColor.a = transpColor.a + (1 - transpColor.a) * nextColor.a;
        }
    }

    vec3 worldPos = texture(gPosition, inUV).rgb;
    float depth = texture(gDepth, inUV).r;

    vec4 solidColor = texture(gAlbedo, inUV);
    vec4 emission = texture(gEmissive, inUV);
    vec3 normal = sphericalAnglesToNormal(texture(gNormal, inUV).xy);

    float shadow = calc_shadow(worldPos, normal);
//    shadow = 1.0 - (1 - shadow) * 0.5;

    vec3 sun_dir = info.main_light_dir.xyz;
    vec3 skyCol = calculateSky(inUV, info.frame_size, info.camera.pos.xyz, info.camera.dir.xyz, info.camera.fovy, info.camera.view, sun_dir);

    float areaLightCosImportance = 0.1;
    float cosFactor = 1 - (1 - dot(normal, -sun_dir)) * areaLightCosImportance;

    // Blend transparent with solid colors
    vec3 currColor = mix(solidColor.rgb * cosFactor, transpColor.rgb, transpColor.a);
    // Apply additional emission
    currColor.rgb += emission.rgb;

    if (depth < 0.0001) {
        skyCol = 1.0 - exp(-2.0 * skyCol);
        outColor = vec4(skyCol, 1);
    } else {
        outColor = vec4(currColor * shadow, 1);
    }
}
