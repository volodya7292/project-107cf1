#include "common.hlsl"

struct Node {
    uint element_id;
    uint child_a;
    uint child_b;
};

struct PushConstants {
    uint morton_codes_offset;
    uint nodes_offset;
    uint n_elements;
};

RWByteAddressBuffer buffer : register(u0);

[[vk::push_constant]]
PushConstants params;

uint2 determineRange(int idx) {
    uint ptr_global = params.morton_codes_offset + idx * sizeof(MortonCode);
    uint curr_code = buffer.Load<MortonCode>(ptr_global).code;

    // determine the range of keys covered by each internal node (as well as its children)
    // direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    // the index is either the beginning of the range or the end of the range
    int common_prefix_with_left = 0;
    int common_prefix_with_right = int(clz(curr_code ^ buffer.Load<MortonCode>(ptr_global + sizeof(MortonCode)).code));

    if (idx == 0) {
        common_prefix_with_left = -1;
    } else {
        common_prefix_with_left = int(clz(curr_code ^ buffer.Load<MortonCode>(ptr_global - sizeof(MortonCode)).code));
    }

    int direction = common_prefix_with_right > common_prefix_with_left ? 1 : -1;
    int min_prefix_range = 0;

    if (idx == 0) {
        min_prefix_range = -1;
    } else {
        min_prefix_range = int(clz(curr_code ^ buffer.Load<MortonCode>(ptr_global - direction * sizeof(MortonCode)).code));
    }

    int lmax = 2;
    int next_key = idx + lmax * direction;
    uint next_key_gptr = params.morton_codes_offset + next_key * sizeof(MortonCode);

    while (next_key >= 0 && next_key < params.n_elements && clz(curr_code ^ buffer.Load<MortonCode>(next_key_gptr).code) > min_prefix_range) {
        lmax *= 2;
        next_key = idx + lmax * direction;
        next_key_gptr = params.morton_codes_offset + next_key * sizeof(MortonCode);
    }

    //find the other end using binary search
    int l = 0;

    do {
        lmax = (lmax + 1) >> 1;// exponential decrease
        int new_val = idx + (l + lmax) * direction;

        if (new_val >= 0 && new_val < params.n_elements) {
            uint code = buffer.Load<MortonCode>(params.morton_codes_offset + new_val * sizeof(MortonCode)).code;
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

uint findSplit(uint first, uint last) {
    // Identical Morton codes => split the range in the middle.
    uint firstCode = buffer.Load<MortonCode>(params.morton_codes_offset + first * sizeof(MortonCode)).code;
    uint lastCode = buffer.Load<MortonCode>(params.morton_codes_offset + last * sizeof(MortonCode)).code;

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
            uint splitCode = buffer.Load<MortonCode>(params.morton_codes_offset + newSplit * sizeof(MortonCode)).code;
            uint splitPrefix = clz(firstCode ^ splitCode);

            if (splitPrefix > commonPrefix) {
                // accept proposal
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

[numthreads(THREAD_GROUP_1D_WIDTH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadId) {
    uint idx = DTid.x;
    uint max_nodes = params.n_elements * 2 - 1;// `n_elements` leafs and `n_elements - 1` internal nodes.
    Node node;

    // TODO: fix load/store offsets

    if (idx < params.n_elements) {
        // Leaf node
        node.element_id = buffer.Load<MortonCode>(params.morton_codes_offset + idx * sizeof(MortonCode)).element_id;
        buffer.Store(params.nodes_offset + idx * sizeof(Node), node);
    } else if (idx < max_nodes) {
        // Internal node
        uint internal_idx = idx - params.n_elements;
        uint2 range = determineRange(internal_idx);
        uint split = findSplit(range[0], range[1]);

        if (split == range[0]) {
            // child is a leaf node
            node.child_a = split;
        } else {
            // child is an internal node
            node.child_a = params.n_elements + split;
        }

        if (split + 1 == range[1]) {
            // child is a leaf node
            node.child_b = split + 1;
        } else {
            // child is an internal node
            node.child_b = params.n_elements + split + 1;
        }

        node.element_id = -1;
        buffer.Store(params.nodes_offset + idx * sizeof(Node), node);
    }
}
