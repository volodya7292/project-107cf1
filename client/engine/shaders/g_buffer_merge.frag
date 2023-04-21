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

layout(binding = 6, scalar) uniform FrameInfoBlock {
    FrameInfo info;
};

layout(binding = 7, std430) readonly buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(binding = 8, rgba8) uniform image2DArray translucencyColorsArray;

layout(binding = 9) uniform sampler2DArray mainShadowMap;

layout(binding = 10, scalar) uniform MainShadowInfoBlock {
    float cascadeSplits[MAIN_SHADOW_MAP_N_CASCADES];
    mat4 cascadeProjView[MAIN_SHADOW_MAP_N_CASCADES];
    vec4 lightDir;
} mainShadowInfo;

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

const mat4 biasMat = mat4(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0
);

float textureProj(vec4 shadowCoord, vec2 offset, uint cascadeIndex) {
	float shadow = 1.0;
	float bias = 0.005;

	if (shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) {
		float dist = texture(mainShadowMap, vec3(shadowCoord.st + offset, cascadeIndex)).r;
		if (shadowCoord.w > 0 && dist < shadowCoord.z - bias) {
			shadow = 0.0;
		}
	}

	return shadow;
}

float calc_shadow(vec3 worldPos, vec3 normal) {
    vec4 viewPos = info.camera.view * vec4(worldPos, 1);

    // Get cascade index for the current fragment's view position
    uint cascadeIndex = 0;
    for(uint i = 0; i < MAIN_SHADOW_MAP_N_CASCADES - 1; ++i) {
        if(viewPos.z < mainShadowInfo.cascadeSplits[i]) {
            cascadeIndex = i + 1;
        }
    }

    vec4 worldPosLightClipSpace = mainShadowInfo.cascadeProjView[cascadeIndex] * vec4(worldPos, 1);
    worldPosLightClipSpace.y = -worldPosLightClipSpace.y;

    vec3 worldPosLightNDC = worldPosLightClipSpace.xyz / worldPosLightClipSpace.w;
    vec2 worldPosLightNorm = worldPosLightNDC.xy * 0.5 + 0.5;
    float depthAtWorldPos = worldPosLightNDC.z;


    float shadowFactor = 0;
    int pfcRange = 1;
    uint count = 0;

    ivec2 texSize = textureSize(mainShadowMap, 0).xy;
    vec2 texelSize = 0.75 / texSize;

    for (int x = -pfcRange; x <= pfcRange; x++) {
    	for (int y = -pfcRange; y <= pfcRange; y++) {
    	    vec2 offset = texelSize * vec2(x, y);
            float shadowMapDepth = texture(mainShadowMap, vec3(worldPosLightNorm.xy + offset, cascadeIndex)).r;

            float bias = max(0.5 * (1.0 - dot(normal, -info.main_light.dir.xyz)), 0.005);

            float shadow = float((depthAtWorldPos + bias) > shadowMapDepth);
            shadowFactor += shadow;
            count += 1;
    	}
    }

    shadowFactor /= count;

    return shadowFactor;
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

    vec3 normal = sphericalAnglesToNormal(texture(gNormal, inUV).xy);

    float shadow = calc_shadow(worldPos, normal);

    outColor = vec4(currColor.rgb * shadow, 1);
}
