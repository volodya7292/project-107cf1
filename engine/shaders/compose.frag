#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 0) uniform sampler2D albedo;

layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 1, scalar) uniform FrameData {
    FrameInfo info;
};

layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 5, std430) coherent buffer TranslucentDepthsArray {
    uint depthsArray[];
};
layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 6, rgba8) uniform image2DArray translucencyColorsArray;
 
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint sliceSize = info.frame_size.x * info.frame_size.y;

    vec4 currColor = vec4(0);

    // Collect translucency
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        if (depthsArray[coordIdx + i * sliceSize] == 0xFFFFFFFFu) {
            // The following layers do not contain any colors, stop the loop
            break;
        } else {
            vec4 nextColor = imageLoad(translucencyColorsArray, ivec3(coord, i));
            // Note: reverse blending
            currColor.rgb = mix(nextColor.rgb, currColor.rgb, currColor.a);
            currColor.a = currColor.a + (1 - currColor.a) * nextColor.a;
        }
    }

    // Blend with solid colors
    vec4 solidColor = texture(albedo, inUV);
    currColor = mix(solidColor, currColor, currColor.a);

    // outColor = texture(albedo, inUV);//  vec4(0, 0.5, 0.1, 1.0);
    outColor = vec4(currColor.rgb, 1);
}
