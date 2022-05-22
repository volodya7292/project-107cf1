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
// Determines the range of morton codes for the specified internal node (idx)
uint2 mortonDetermineRange(int idx, uint morton_codes_offset, uint n_elements) {
    uint ptr_global = morton_codes_offset + idx * sizeof(MortonCode);
    uint curr_code = mem.Load<MortonCode>(ptr_global).code;

    // determine the range of keys covered by each internal node (as well as its children)
    // direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    // the index is either the beginning of the range or the end of the range
    int common_prefix_with_left = 0;
    int common_prefix_with_right = int(clz(curr_code ^ mem.Load<MortonCode>(ptr_global + sizeof(MortonCode)).code));

    if (idx == 0) {
        common_prefix_with_left = -1;
    } else {
        common_prefix_with_left = int(clz(curr_code ^ mem.Load<MortonCode>(ptr_global - sizeof(MortonCode)).code));
    }

    int direction = common_prefix_with_right > common_prefix_with_left ? 1 : -1;
    int min_prefix_range = 0;

    if (idx == 0) {
        min_prefix_range = -1;
    } else {
        min_prefix_range = int(clz(curr_code ^ mem.Load<MortonCode>(ptr_global - direction * sizeof(MortonCode)).code));
    }

    int lmax = 2;
    int next_key = idx + lmax * direction;
    uint next_key_gptr = morton_codes_offset + next_key * sizeof(MortonCode);

    while (next_key >= 0 && next_key < n_elements && clz(curr_code ^ mem.Load<MortonCode>(next_key_gptr).code) > min_prefix_range) {
        lmax *= 2;
        next_key = idx + lmax * direction;
        next_key_gptr = morton_codes_offset + next_key * sizeof(MortonCode);
    }

    //find the other end using binary search
    int l = 0;

    do {
        lmax = (lmax + 1) >> 1;// exponential decrease
        int new_val = idx + (l + lmax) * direction;

        if (new_val >= 0 && new_val < n_elements) {
            uint code = mem.Load<MortonCode>(morton_codes_offset + new_val * sizeof(MortonCode)).code;
            int prefix = int(clz(curr_code ^ code));

            if (prefix > min_prefix_range) {
                l = l + lmax;
            }
        }
    } while (lmax > 1);

    int j = idx + l * direction;
    int left = 0;
    int right = 0;

    if (idx < j) {
        left = idx;
        right = j;
    } else {
        left = j;
        right = idx;
    }

    return uint2(left, right);
}

// Finds split between two internal nodes using morton codes
uint mortonFindSplit(uint first, uint last, uint morton_codes_offset) {
    // Identical Morton codes => split the range in the middle.
    uint firstCode = mem.Load<MortonCode>(morton_codes_offset + first * sizeof(MortonCode)).code;
    uint lastCode = mem.Load<MortonCode>(morton_codes_offset + last * sizeof(MortonCode)).code;

    if (firstCode == lastCode) {
        return (first + last) >> 1;
    }

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros.
    uint commonPrefix = clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    uint split = first;// initial guess
    uint step = last - first;

    do {
        step = (step + 1) >> 1;// exponential decrease
        uint newSplit = split + step;// proposed new position

        if (newSplit < last) {
            uint splitCode = mem.Load<MortonCode>(morton_codes_offset + newSplit * sizeof(MortonCode)).code;
            uint splitPrefix = clz(firstCode ^ splitCode);

            if (splitPrefix > commonPrefix) {
                // accept proposal
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}
#endif