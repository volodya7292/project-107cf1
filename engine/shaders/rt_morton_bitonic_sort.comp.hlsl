#include "morton.hlsli"

// https://en.wikipedia.org/wiki/Bitonic_sorter#Alternative_representation

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

groupshared MortonCode local_value[WORK_GROUP_SIZE * 2];

bool compare_fn(in const uint left, in const uint right) {
    return left > right;
}

void global_compare_and_swap(uint2 idx) {
    if (any(idx >= params.n_values)) {
        return;
    }

    uint2 gptr = params.data_offset.xx + idx * sizeof(MortonCode);
    MortonCode valX = buffer.Load<MortonCode>(gptr.x);
    MortonCode valY = buffer.Load<MortonCode>(gptr.y);

    if (compare_fn(valX.code, valY.code)) {
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

void big_flip(uint h, uint t_prime) {
    uint half_h = h >> 1;

    uint q = ((2 * t_prime) / h) * h;
    uint tm = (t_prime % half_h);
    uint2 indices = q.xx + uint2(tm, h-1 - tm);

    global_compare_and_swap(indices);
}

void big_disperse(uint h, uint t_prime) {
    uint half_h = h >> 1;

    uint q = ((2 * t_prime) / h) * h;
    uint tm = t_prime % half_h;
    uint2 indices = uint2(tm, tm + half_h) + q;

    global_compare_and_swap(indices);
}

void local_flip(uint h, uint t, uint offset) {
    GroupMemoryBarrierWithGroupSync();

    uint half_h = h >> 1;
    uint q = ((2 * t) / h) * h;
    uint tm = t % half_h;
    uint2 indices = q.xx + uint2(tm, h-1 - tm);

    if (all(indices + offset < params.n_values)) {
        local_compare_and_swap(indices);
    }
}

void local_disperse(uint h, uint t, uint offset) {
    for (; h > 1; h /= 2) {
        GroupMemoryBarrierWithGroupSync();

        uint half_h = h >> 1;
        uint q = ((2 * t) / h) * h;
        uint tm = t % half_h;
        uint2 indices = q.xx + uint2(tm, half_h + tm);

        if (all(indices + offset < params.n_values)) {
            local_compare_and_swap(indices);
        }
    }
}

void local_bitonic_merge_sort(uint h, uint t, uint offset) {
    for (uint hh = 2; hh <= h; hh <<= 1) {
        local_flip(hh, t, offset);
        local_disperse(hh/2, t, offset);
    }
}

[numthreads(WORK_GROUP_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupId) {
    uint t_prime = DTid.x;
    uint t = GTid.x;
    uint offset = WORK_GROUP_SIZE * 2 * Gid.x;// we can use offset if we have more than one invocation.

    if (params.algorithm <= eLocalDisperse) {
        if (offset+t*2 < params.n_values) {
            local_value[t*2] = buffer.Load<MortonCode>(params.data_offset + (offset+t*2) * sizeof(MortonCode));
        }
        if (offset+t*2+1 < params.n_values) {
            local_value[t*2+1] = buffer.Load<MortonCode>(params.data_offset + (offset+t*2+1) * sizeof(MortonCode));
        }
    }

    switch (params.algorithm) {
        case eLocalBitonicMergeSort:
        local_bitonic_merge_sort(params.h, t, offset);
        break;
        case eLocalDisperse:
        local_disperse(params.h, t, offset);
        break;
        case eBigFlip:
        big_flip(params.h, t_prime);
        break;
        case eBigDisperse:
        big_disperse(params.h, t_prime);
        break;
    }

    if (params.algorithm <= eLocalDisperse) {
        GroupMemoryBarrierWithGroupSync();

        if (offset+t*2 < params.n_values) {
            buffer.Store(params.data_offset + (offset+t*2) * sizeof(MortonCode), local_value[t*2]);
        }
        if (offset+t*2+1 < params.n_values) {
            buffer.Store(params.data_offset + (offset+t*2+1) * sizeof(MortonCode), local_value[t*2+1]);
        }
    }
}