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

template<typename T>
struct SubGlobalBuffer {
    RWByteAddressBuffer buffer;
    uint offset;

    T Load(uint index) {
        return buffer.Load<T>(offset + index * sizeof(T));
    }

    void Store(uint index, T item) {
        return buffer.Store(offset + index * sizeof(T), item);
    }
};
