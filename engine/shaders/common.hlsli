#define THREAD_GROUP_WIDTH 8
#define THREAD_GROUP_HEIGHT 8
#define THREAD_GROUP_1D_WIDTH 64

// Counts the number of leading zeros
uint clz(uint v) {
    return 31 - firstbithigh(v);
}

struct MortonCode {
    uint code;
    uint element_id;
};

struct Bounds {
    float3 p_min;
    float3 p_max;
};

#define LBVHNode_bounds_offset 0

struct LBVHNode {
    Bounds bounds;
    uint element_index;
    uint parent;
    uint child_a;
    uint child_b;
};

template<typename T>
struct SubGlobalBuffer {
    RWByteAddressBuffer buffer;
    uint offset;

    T Load(uint index) {
        return buffer.Load<T>(offset + index * sizeof(T));
    }

    void Store(uint index, T item) {
        buffer.Store(offset + index * sizeof(T), item);
    }

    template<typename S>
    void StoreWithOffset(uint index, uint byte_offset, S item) {
        buffer.Store(offset + index * sizeof(T) + byte_offset, item);
    }
};
