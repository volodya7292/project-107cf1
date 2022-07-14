#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(set = 0, binding = 5, std430) coherent buffer DepthsArray {
    uint depthsArray[];
};

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    uvec2 frameSize;
};

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint currDepth = floatBitsToUint(gl_FragCoord.z);
    uint coordIdx = frameSize.x * coord.y + coord.x;
    uint sliceSize = frameSize.x * frameSize.y;

    // Early depth test
    uint lastDepth = depthsArray[coordIdx + (OIT_N_CLOSEST_LAYERS - 1) * sliceSize];
    if (currDepth > lastDepth) {
        return;
    }

    // Find OIT_N_CLOSEST_LAYERS minimum depths
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        uint prev = atomicMin(depthsArray[coordIdx + i * sliceSize], currDepth);
        if (prev == 0xFFFFFFFFu || prev == currDepth) {
            break;
        }
        currDepth = max(prev, currDepth);
    }
}