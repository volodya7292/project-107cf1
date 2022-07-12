#version 450

#define N_LAYERS 4

layout(std430, set = 0, binding = 5) coherent buffer DepthsArray {
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
    uint lastDepth = depthsArray[coordIdx + (N_LAYERS - 1) * sliceSize];
    if (currDepth > lastDepth) {
        return;
    }

    // Find N_LAYERS minimum depths
    for (uint i = 0; i < N_LAYERS; i++) {
        uint prev = atomicMin(depthsArray[coordIdx + i * sliceSize], currDepth);
        if (prev == 0xFFFFFFFFu || prev == currDepth) {
            break;
        }
        currDepth = max(prev, currDepth);
    }
}