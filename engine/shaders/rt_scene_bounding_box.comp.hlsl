#include "common.hlsli"

#define WORK_GROUP_SIZE 512

struct PushConstants {
	uint top_nodes_offset;
	uint temp_aabbs_offset;
    uint n_elements;
	uint iteration;
};  

RWByteAddressBuffer buffer : register(u0);

[[vk::push_constant]]
PushConstants params;

// Shared memory limit is 16384 bytes; WORK_GROUP_SIZE (512) * sizeof(Bounds) = 12288 bytes
groupshared Bounds group_bounds[WORK_GROUP_SIZE]; 

[numthreads(WORK_GROUP_SIZE, 1, 1)]
void main(uint3 Gid : SV_GroupId, uint3 GTid : SV_GroupThreadId) {
    uint t = GTid.x;
	uint group_id = Gid.x;
    uint group_start = group_id * WORK_GROUP_SIZE;

    if (group_start + t*2+1 >= params.n_elements) {
        return;
    }

	// Leaves go after internal nodes. 
    // The number of internal nodes are one less than the number of leaves.
    uint nodes_leaves_offset = params.top_nodes_offset + (params.n_elements - 1) * sizeof(TopLBVHNode); 

	uint global_step = pow(WORK_GROUP_SIZE, params.iteration);

	// Load bounds into groupshared memory
	if (params.iteration == 0) {
 	   	Bounds b0 = buffer.Load<TopLBVHNode>(nodes_leaves_offset + (group_start + t*2) * sizeof(TopLBVHNode)).bounds;
    	Bounds b1 = buffer.Load<TopLBVHNode>(nodes_leaves_offset + (group_start + t*2+1) * sizeof(TopLBVHNode)).bounds;
		group_bounds[t] = b0.combine(b1);
	} else {
		Bounds b0 = buffer.Load<Bounds>(params.temp_aabbs_offset + (group_start + t*2) * global_step * sizeof(Bounds));
		Bounds b1 = buffer.Load<Bounds>(params.temp_aabbs_offset + (group_start + t*2+1) * global_step * sizeof(Bounds));
		group_bounds[t] = b0.combine(b1);
	}

	const uint max_iters = log2(WORK_GROUP_SIZE);

	// Recursively combile all aabbs in the group
	[loop]
    for (uint i = 0; i < max_iters; i++) {
		uint step = 1u << i;
		if (t % (step * 2) != 0) {
			// Terminate unnecessary threads
			return;
		}

        GroupMemoryBarrierWithGroupSync();

		Bounds sb0 = group_bounds[t];
		Bounds sb1 = group_bounds[t + step];
		group_bounds[t] = sb0.combine(sb1);
    }

	buffer.Store(params.temp_aabbs_offset + group_id * WORK_GROUP_SIZE * global_step * sizeof(Bounds), group_bounds[0]);
}