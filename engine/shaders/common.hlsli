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

// Calculates center point (within the unit cube [0,1]) of an inner AABB inside outer AABB
// cmin, cmax: outer AABB
// min, max: inner AABB
inline float3 aabbComputeCenter(float3 cmin, float3 cmax, float3 min, float3 max) {
	float3 len = cmax - cmin;

	float3 tmpMin = (min - cmin) / len;
	float3 tmpMax = (max - cmin) / len;

	float3 axis = tmpMax - tmpMin;
	float d = length(axis) * 0.5f;

    // min + half of bound size
	return tmpMin + d * normalize(axis); 
}

// ----------------------------------------------------------------------

struct Bounds {
    float3 p_min;
    float3 p_max;

    inline Bounds combine(Bounds other) {
        other.p_min = min(p_min, other.p_min);
        other.p_max = max(p_max, other.p_max);
        return other;
    }
};

#define LBVHNode_bounds_offset 0
struct LBVHNode {
    Bounds bounds;
    uint element_id;
    uint parent;
    uint child_a;
    uint child_b;
};

#define LBVHInstance_bounds_offset (4 * 3 + 64 * 2)
struct LBVHInstance {
    uint indices_offset;
    uint vertices_offset;
    uint nodes_offset;
    float4x4 transform;
    float4x4 transform_inverse;
    Bounds bounds;
};

#define TopLBVHNode_instance_offset 0
struct TopLBVHNode {
    LBVHInstance instance;
    uint parent;
    uint child_a;
    uint child_b;
};