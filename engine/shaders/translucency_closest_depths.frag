#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../engine/shaders/common.glsl"

layout(early_fragment_tests) in;

layout (input_attachment_index = 0, set = DESC_SET_GENERAL_PER_FRAME, binding = 7) uniform subpassInput inputSolidDepth;

layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 0, scalar) uniform FrameData {
    FrameInfo info;
};
layout(set = DESC_SET_GENERAL_PER_FRAME, binding = 5, std430) coherent buffer DepthsArray {
    uint transparencyDepthsArray[];
};

layout(location = 0) out vec4 outColor;

void main() {
    // Prevent Z-fighting
    {
        float solidDepth = subpassLoad(inputSolidDepth).r;
        solidDepth = linearize_depth(solidDepth, info.camera.z_near);
        float fragDepth = linearize_depth(gl_FragCoord.z, info.camera.z_near);

        // Do not render transparency where depth is uncertain
        if (fragDepth > solidDepth - 0.001 * solidDepth) {
            return;
        }
    }

    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint currDepth = floatBitsToUint(gl_FragCoord.z);
    uint coordIdx = info.frame_size.x * coord.y + coord.x;
    uint sliceSize = info.frame_size.x * info.frame_size.y;

    // Early transparency depth test (check farthest layer)
    uint lastLayerIdx = OIT_N_CLOSEST_LAYERS - 1;
    uint lastDepth = transparencyDepthsArray[coordIdx + lastLayerIdx * sliceSize];
    if (currDepth > lastDepth) {
        return;
    }

    // Find OIT_N_CLOSEST_LAYERS minimum depths
    for (uint i = 0; i < OIT_N_CLOSEST_LAYERS; i++) {
        uint prev = atomicMin(transparencyDepthsArray[coordIdx + i * sliceSize], currDepth);
        if (prev == 0xFFFFFFFFu || prev == currDepth) {
            break;
        }
        currDepth = max(prev, currDepth);
    }
}
