#include "common.hlsli"

// https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

struct MortonCode {
    uint code;
    uint element_id;
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline uint3 mortonExpandBits(uint3 v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
inline uint morton3D(float3 p) {
    uint3 upoint = uint3(clamp(p * 1024.0, 0.0, 1023.0));
    uint3 ex = mortonExpandBits(upoint);
    return ex.x * 4 + ex.y * 2 + ex.z;
}

#ifdef MORTON_LBVH_GEN

// Computes longest commot prefix between two morton codes specified by indices i,j
inline int mortonLCP(uint first, uint last, uint n_elements, uint morton_codes_offset) {
    if (first >= n_elements || last >= n_elements) {
        return -1;
    }

    uint code_a = mem.Load<MortonCode>(morton_codes_offset + first * sizeof(MortonCode)).code;
    uint code_b = mem.Load<MortonCode>(morton_codes_offset + last * sizeof(MortonCode)).code;

	if (code_a != code_b) {
        return clz(code_a ^ code_b);
	} else {
        // TODO: Technically this should be primitive ID (code_a.element_id)

        // Morton codes are equal, hence use primitive ids
        // Note: Add 31 because clz(code_a ^ code_b) = 31 (see "concatenation" in the paper).
        return clz(first ^ last) + 31;
	}

}

// Determines the range of morton codes for the specified internal node (idx)
uint2 mortonDetermineRange(uint idx, uint n_elements, uint morton_codes_offset) {
    uint ne = n_elements;
    uint mo = morton_codes_offset;

    int d = mortonLCP(idx, idx + 1, ne, mo) - mortonLCP(idx, idx - 1, ne, mo);
    d = clamp(d, -1, 1);
    int minPrefix = mortonLCP(idx, idx - d, ne, mo);

    int maxLength = 2;
    while (mortonLCP(idx, idx + maxLength * d, ne, mo) > minPrefix) {
        maxLength *= 4;
    }

    int length = 0;
    for (int t = maxLength / 2; t > 0; t /= 2) {
        if (mortonLCP(idx, idx + (length + t) * d, ne, mo) > minPrefix) {
            length = length + t;
        }
    }

    int j = idx + length * d;
    return uint2(min(idx, j), max(idx, j));
}

// Finds split between two internal nodes using morton codes
int mortonFindSplit(int first, uint last, uint n_elements, uint morton_codes_offset) {
    int commonPrefix = mortonLCP(first, last, n_elements, morton_codes_offset);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = mortonLCP(first, newSplit, n_elements, morton_codes_offset);
            if (splitPrefix > commonPrefix)
                split = newSplit;
        }
    } while (step > 1);

    return split;
}

#endif