#include "morton.hlsli"

// This shader implements a sorting network.
// It is follows the alternative notation for bitonic sorting networks, as given at:
// https://en.m.wikipedia.org/wiki/Bitonic_sorter#Alternative_representation

// Note: there exist hardware limits
// sizeof(local_value[]) : Must be <= maxComputeSharedMemorySize
// local_size_x          : Must be <= maxComputeWorkGroupInvocations


#define eLocalBitonicMergeSort 0
#define eLocalDisperse 1
#define eBigFlip 2
#define eBigDisperse 3

#define WORK_GROUP_SIZE 1024

struct PushConstants {
    uint h;
    uint algorithm;
    uint n_values;
    uint data_offset;
};

RWByteAddressBuffer buffer : register(u0);

[[vk::push_constant]]
PushConstants params;

// Workgroup local memory. We use this to minimise round-trips to global memory.
// It allows us to evaluate a sorting network of up to 1024 with one shader invocation.
groupshared MortonCode local_value[WORK_GROUP_SIZE * 2];

// naive comparison
bool compare_fn(in const uint left, in const uint right) {
    return left > right;
}

void global_compare_and_swap(uint2 idx) {
    uint2 gptr = params.data_offset.xx + idx * sizeof(MortonCode);
    MortonCode valX = buffer.Load<MortonCode>(gptr.x);
    MortonCode valY = buffer.Load<MortonCode>(gptr.y);

    if (idx.x < params.n_values && idx.y < params.n_values && compare_fn(valX.code, valY.code)) {
        buffer.Store(gptr.x, valY);
        buffer.Store(gptr.y, valX);
    }
}

void local_compare_and_swap(uint2 idx) {
    if (compare_fn(local_value[idx.x].code, local_value[idx.y].code)) {
        MortonCode tmp = local_value[idx.x];
        local_value[idx.x] = local_value[idx.y];
        local_value[idx.y] = tmp;
    }
}

// Performs full-height flip (h height) over globally available indices.
void big_flip(in uint h, uint t_prime : SV_DispatchThreadID) {
    uint half_h = h >> 1;// Note: h >> 1 is equivalent to h / 2

    uint q = ((2 * t_prime) / h) * h;
    uint x = q     + (t_prime % half_h);
    uint y = q + h - (t_prime % half_h) - 1;

    global_compare_and_swap(uint2(x, y));
}

// Performs full-height disperse (h height) over globally available indices.
void big_disperse(in uint h, uint t_prime : SV_DispatchThreadID) {
    uint half_h = h >> 1;// Note: h >> 1 is equivalent to h / 2

    uint q = ((2 * t_prime) / h) * h;
    uint x = q + (t_prime % (half_h));
    uint y = q + (t_prime % (half_h)) + half_h;

    global_compare_and_swap(uint2(x, y));
}

// Performs full-height flip (h height) over locally available indices.
void local_flip(in uint h, in uint t : SV_GroupThreadID) {
    GroupMemoryBarrierWithGroupSync();

    uint half_h = h >> 1;// Note: h >> 1 is equivalent to h / 2
    uint2 indices = (h * ((2 * t) / h)).xx + int2(t % half_h, h - 1 - (t % half_h));

    local_compare_and_swap(indices);
}

// Performs progressively diminishing disperse operations (starting with height h)
// on locally available indices: e.g. h==8 -> 8 : 4 : 2.
// One disperse operation for every time we can divide h by 2.
void local_disperse(in uint h, in uint t : SV_GroupThreadID) {
    for (; h > 1; h /= 2) {
        GroupMemoryBarrierWithGroupSync();

        uint half_h = h >> 1;// Note: h >> 1 is equivalent to h / 2
        uint2 indices = (h * ((2 * t) / h)).xx + int2(t % half_h, half_h + (t % half_h));

        local_compare_and_swap(indices);
    }
}

void local_bitonic_merge_sort(uint h, in uint t : SV_GroupThreadID) {
    for (uint hh = 2; hh <= h; hh <<= 1) { // note:  h <<= 1 is same as h *= 2
        local_flip(hh, t);
        local_disperse(hh/2, t);
    }
}

[numthreads(WORK_GROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupId) {
    uint t_prime = DTid.x;
    uint t = GTid.x;
    uint offset = WORK_GROUP_SIZE * 2 * Gid.x;// we can use offset if we have more than one invocation.

    if (params.algorithm <= eLocalDisperse) {
        // In case this shader executes a `local_` algorithm, we must
        // first populate the workgroup's local memory.
        if (offset+t*2+1 >= params.n_values) {
            return;
        }
        local_value[t*2]   = buffer.Load<MortonCode>(params.data_offset + offset+t*2 * sizeof(MortonCode));
        local_value[t*2+1] = buffer.Load<MortonCode>(params.data_offset + offset+t*2+1 * sizeof(MortonCode));
    }

    switch (params.algorithm) {
        case eLocalBitonicMergeSort:
        local_bitonic_merge_sort(params.h, t);
        break;
        case eLocalDisperse:
        local_disperse(params.h, t);
        break;
        case eBigFlip:
        big_flip(params.h, t_prime);
        break;
        case eBigDisperse:
        big_disperse(params.h, t_prime);
        break;
    }

    // Write local memory back to buffer in case we pulled in the first place.
    if (params.algorithm <= eLocalDisperse) {
        GroupMemoryBarrierWithGroupSync();
        // push to global memory
        buffer.Store(params.data_offset + offset+t*2 * sizeof(MortonCode), local_value[t*2]);
        buffer.Store(params.data_offset + offset+t*2+1 * sizeof(MortonCode), local_value[t*2+1]);
    }
}