#define THREAD_GROUP_WIDTH 8
#define THREAD_GROUP_HEIGHT 8
#define THREAD_GROUP_1D_WIDTH 64

#define FLT_MAX 3.402823466e+38f

// Counts the number of leading zeros
uint clz(uint v) {
    return 31 - firstbithigh(v);
}

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

inline float3 mortonComputeCenter(float3 cmin, float3 cmax, float3 min, float3 max) {
	float3 len = cmax - cmin;

	float3 tmpMin = (min - cmin) / len;
	float3 tmpMax = (max - cmin) / len;

	float3 axis = tmpMax - tmpMin;
	float d = length(axis) * 0.5f;

    // min + half of bound size
	return tmpMin + d * normalize(axis); 
}

// ----------------------------------------------------------------------

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

struct LBVHInstance {
    uint indices_offset;
    uint vertices_offset;
    uint nodes_offset;
    float4x4 transform;
    float4x4 transform_inverse;
};

#define TopLBVHNode_bounds_offset sizeof(LBVHInstance)
struct TopLBVHNode {
    LBVHInstance instance;
    Bounds bounds;
    uint parent;
    uint child_a;
    uint child_b;
};