#define THREAD_GROUP_WIDTH 8
#define THREAD_GROUP_HEIGHT 8
#define THREAD_GROUP_1D_WIDTH 64

#define FLT_MAX 3.402823466e+38f

// Counts the number of leading zeros
inline uint clz(uint v) {
    return 31 - firstbithigh(v);
}

inline uint div_ceil(uint a, uint b) {
    return (a + b - 1) / b;
}

template<typename T>
struct SubGlobalBuffer {
    RWByteAddressBuffer buffer;
    uint offset;

    inline T Load(uint index) {
        return buffer.Load<T>(offset + index * sizeof(T));
    }

    template<typename S>
    inline S LoadWithOffset(uint index, uint byte_offset) {
        return buffer.Load<S>(offset + index * sizeof(T) + byte_offset);
    }

    inline void Store(uint index, T item) {
        buffer.Store(offset + index * sizeof(T), item);
    }

    template<typename S>
    inline void StoreWithOffset(uint index, uint byte_offset, S item) {
        buffer.Store(offset + index * sizeof(T) + byte_offset, item);
    }

    inline uint AtomicAdd(uint index, uint byte_offset, uint value) {
        uint original;
        buffer.InterlockedAdd(offset + index * sizeof(T) + byte_offset, value, original);
        return original;
    }

    inline uint AtomicLoad(uint index, uint byte_offset) {
        return AtomicAdd(index, byte_offset, 0);
    }

    inline uint AtomicExchange(uint index, uint byte_offset, uint value) {
        uint original;
        buffer.InterlockedExchange(offset + index * sizeof(T) + byte_offset, value, original);
        return original;
    }
};

// Calculates center point (within the unit cube [0,1]) of an inner AABB inside outer AABB
// cmin, cmax: outer AABB
// min, max: inner AABB
inline float3 aabbComputeCenter(float3 cmin, float3 cmax, float3 min, float3 max) {
	float3 len = cmax - cmin;

    for (uint i = 0; i < 3; i++) {
        if (len[i] == 0)
            len[i] = 1;
    }

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
#define LBVHNode_element_id_offset sizeof(Bounds)
#define LBVHNode_parent_offset (sizeof(Bounds) + 4)
#define LBVHNode_child_a_offset (sizeof(Bounds) + 8)
#define LBVHNode_child_b_offset (sizeof(Bounds) + 12)
struct LBVHNode {
    Bounds bounds;
    uint element_id;
    uint parent;
    uint child_a;
    uint child_b;
};

#define LBVHInstance_nodes_offset_offset (sizeof(uint) * 2)
#define LBVHInstance_bounds_offset (sizeof(uint) * 3 + sizeof(float4x4) * 2)
struct LBVHInstance {
    uint indices_offset;
    uint vertices_offset;
    uint nodes_offset;
    float4x4 transform;
    float4x4 transform_inverse;
    Bounds bounds;
};

#define TopLBVHNode_instance_offset 0
#define TopLBVHNode_parent_offset sizeof(LBVHInstance)
#define TopLBVHNode_child_a_offset (sizeof(LBVHInstance) + sizeof(uint))
#define TopLBVHNode_child_b_offset (sizeof(LBVHInstance) + sizeof(uint) * 2)
struct TopLBVHNode {
    LBVHInstance instance;
    uint parent;
    uint child_a;
    uint child_b;
};