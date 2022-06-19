#include "common.hlsli"

#define RT_EPSILON  1e-8
#define RT_RAY_BIAS 1e-4f

#define INTERSECTION_FLAG_NONE        0
// #define INTERSECTION_FLAG_NORMAL      1 << 0
#define INTERSECTION_FLAG_TEXCOORD    1 << 1

// Maximum number of triangles in a bottom LBVH = 1024^2 = 2^10
static uint rt_bottom_traversal_stack[32];
static uint rt_top_traversal_stack[32];

struct TriInterInput {
    float3 ray_orig;
    float3 ray_dir;
    uint indices_offset;
    uint vertices_offset;
    uint tex_coords_offset;
    uint triangle_id;
    uint flags;
};

struct TriangleIntersection {
	bool   intersected;
	uint   triangle_id;
	float3 inter_point;
	float3 normal;
	float2 tex_coord;
	float  distance;
};

TriangleIntersection rt_intersect_triangle(in TriInterInput input) {
    TriangleIntersection inter;
    inter.intersected = false;
    inter.distance = FLT_MAX;
    inter.triangle_id = input.triangle_id;

    SubGlobalBuffer<uint3> indices = {mem, input.indices_offset};
    SubGlobalBuffer<float3> vertices = {mem, input.vertices_offset};

    uint3 tri_indices = indices.Load(inter.triangle_id);
    float3 v0 = vertices.Load(tri_indices[0]);
    float3 v1 = vertices.Load(tri_indices[1]);
    float3 v2 = vertices.Load(tri_indices[2]);

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 pvec = cross(input.ray_dir, edge2);
    float det = dot(edge1, pvec);

    if (det > 0)
        // if the determinant is negative, the triangle is backfacing
        return inter;

    // Note: use this when backfacing culling is disabled
    // if (abs(det) < RT_EPSILON)
    //     // if the determinant is 0, the ray misses the triangle
    //     return inter;
    
    float inv_det = 1.0 / det;
    
    float3 tvec = input.ray_orig - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1)
        return inter;
    
    float3 qvec = cross(tvec, edge1);
    float v = dot(input.ray_dir, qvec) * inv_det;
    if (v < 0 || u + v > 1)
        return inter;

    float t = dot(edge2, qvec) * inv_det;
    if (t < 0)
        return inter;

    inter.intersected = true;
    inter.distance = t;
    inter.inter_point = input.ray_orig + input.ray_dir * t;
    inter.normal = normalize(cross(edge1, edge2));

    if (input.flags & INTERSECTION_FLAG_TEXCOORD) {
        SubGlobalBuffer<float2> tex_coords = {mem, input.tex_coords_offset};

        float2 tc0 = tex_coords.Load(tri_indices[0]);
        float2 tc1 = tex_coords.Load(tri_indices[1]);
        float2 tc2 = tex_coords.Load(tri_indices[2]);

        inter.tex_coord = tc0 * (1.0 - u - v) + tc1 * u + tc2 * v;
    }

    return inter;
}

TriangleIntersection rt_intersect_bottom_lbvh(float3 ray_orig, float3 ray_dir, LBVHInstance instance) {
    TriangleIntersection min_inter;
    min_inter.intersected = false;
    min_inter.distance = FLT_MAX;

    SubGlobalBuffer<LBVHNode> nodes = {mem, instance.nodes_offset};

    // ray_orig = mul(instance.transform_inverse, float4(ray_orig, 1)).xyz;
    // ray_dir = mul(instance.transform_inverse, float4(ray_dir, 0)).xyz;
    ray_orig = mul(float4(ray_orig, 1), instance.transform_inverse).xyz;
    ray_dir = mul(float4(ray_dir, 0), instance.transform_inverse).xyz;

    rt_bottom_traversal_stack[0] = -1;

    uint stack_index = 1;
    uint curr_node_id = 0;

    LBVHNode curr_node = nodes.Load(curr_node_id);

    TriInterInput tri_inter_in;
    tri_inter_in.ray_orig = ray_orig;
    tri_inter_in.ray_dir = ray_dir;
    tri_inter_in.indices_offset = instance.indices_offset;
    tri_inter_in.vertices_offset = instance.vertices_offset;

    if (!curr_node.bounds.intersect_ray(ray_orig, ray_dir)) {
        return min_inter;
    }
    if (curr_node.element_id != -1) {
        tri_inter_in.triangle_id = curr_node.element_id;
        TriangleIntersection inter = rt_intersect_triangle(tri_inter_in);
        if (inter.intersected)
            min_inter = inter;
    }

    [loop]
    while (true) {
        bool traverse_a = false;
        bool traverse_b = false;
        uint child_a_id = curr_node.child_a;
        uint child_b_id = curr_node.child_b;

        if (child_a_id != -1) {
            LBVHNode node_a = nodes.Load(child_a_id);

            if (node_a.bounds.intersect_ray(ray_orig, ray_dir)) {
                if (node_a.element_id != -1) {
                    tri_inter_in.triangle_id = node_a.element_id;
                    TriangleIntersection inter = rt_intersect_triangle(tri_inter_in);

                    if (inter.intersected && inter.distance < min_inter.distance) {
                        min_inter = inter;
                    }
                } else {
                    traverse_a = true;
                }
            }
        }

        if (child_b_id != -1) {
            LBVHNode node_b = nodes.Load(child_b_id);

            if (node_b.bounds.intersect_ray(ray_orig, ray_dir)) {
                if (node_b.element_id != -1) {
                    tri_inter_in.triangle_id = node_b.element_id;
                    TriangleIntersection inter = rt_intersect_triangle(tri_inter_in);

                    if (inter.intersected && inter.distance < min_inter.distance) {
                        min_inter = inter;
                    }
                } else {
                    traverse_b = true;
                }
            }
        }

        if (!traverse_a && !traverse_b) {
            // pop from stack
            curr_node_id = rt_bottom_traversal_stack[--stack_index];
        } else {
            curr_node_id = traverse_a ? child_a_id : child_b_id;

            if (traverse_a && traverse_b) {
                // push to stack
                rt_bottom_traversal_stack[stack_index++] = child_b_id;
            }
        }

        if (curr_node_id != -1) {
            curr_node = nodes.Load(curr_node_id); 
        } else {
            break;
        }
    }


    return min_inter;
}

TriangleIntersection trace_ray(float3 ray_orig, float3 ray_dir, uint rt_top_nodes_offset) {
    TriangleIntersection min_inter;
    min_inter.intersected = false;
    min_inter.distance = FLT_MAX;
    float4x4 inter_mesh_transform;

    SubGlobalBuffer<TopLBVHNode> top_nodes = {mem, rt_top_nodes_offset};

    ray_orig += ray_dir * RT_RAY_BIAS;
    rt_top_traversal_stack[0] = -1;

    uint stack_index = 1;
    uint curr_node_id = 0;

    TopLBVHNode curr_node = top_nodes.Load(curr_node_id);

    if (!curr_node.instance.bounds.intersect_ray(ray_orig, ray_dir)) {
        return min_inter;
    }
    if (curr_node.instance.nodes_offset != -1) {
        TriangleIntersection inter = rt_intersect_bottom_lbvh(ray_orig, ray_dir, curr_node.instance);
        if (inter.intersected) {
            min_inter = inter;
            inter_mesh_transform = curr_node.instance.transform;
        }
    }

    [loop]
    while (true) {
        bool traverse_a = false;
        bool traverse_b = false;
        uint child_a_id = curr_node.child_a;
        uint child_b_id = curr_node.child_b;

        if (child_a_id != -1) {
            TopLBVHNode node_a = top_nodes.Load(child_a_id);

            if (node_a.instance.bounds.intersect_ray(ray_orig, ray_dir)) {
                uint nodes_offset = node_a.instance.nodes_offset;

                if (nodes_offset != -1) {
                    TriangleIntersection inter = rt_intersect_bottom_lbvh(ray_orig, ray_dir, node_a.instance);

                    if (inter.distance < min_inter.distance) {
                        min_inter = inter;
                        inter_mesh_transform = node_a.instance.transform;
                    }
                } else {
                    traverse_a = true;
                }
            }
        }

        if (child_b_id != -1) {
            TopLBVHNode node_b = top_nodes.Load(child_b_id);

            if (node_b.instance.bounds.intersect_ray(ray_orig, ray_dir)) {
                uint nodes_offset = node_b.instance.nodes_offset;

                if (nodes_offset != -1) {
                    TriangleIntersection inter = rt_intersect_bottom_lbvh(ray_orig, ray_dir, node_b.instance);

                    if (inter.distance < min_inter.distance) {
                        min_inter = inter;
                        inter_mesh_transform = node_b.instance.transform;
                    }
                } else {
                    traverse_b = true;
                }
            }
        }

        if (!traverse_a && !traverse_b) {
            curr_node_id = rt_top_traversal_stack[--stack_index];
        } else {
            curr_node_id = traverse_a ? child_a_id : child_b_id;

            if (traverse_a && traverse_b) {
                // push to stack
                rt_top_traversal_stack[stack_index++] = child_b_id;
            }
        }

        if (curr_node_id != -1) {
            curr_node = top_nodes.Load(curr_node_id); 
        } else {
            break;
        }
    }

    if (min_inter.intersected) {
        // min_inter.normal = mul(inter_mesh_transform, float4(min_inter.normal, 0)).xyz;
        // min_inter.inter_point = mul(inter_mesh_transform, float4(min_inter.inter_point, 1)).xyz;
        min_inter.normal = mul(float4(min_inter.normal, 0), inter_mesh_transform).xyz;
        min_inter.inter_point = mul(float4(min_inter.inter_point, 1), inter_mesh_transform).xyz;
    }

    return min_inter;
}

