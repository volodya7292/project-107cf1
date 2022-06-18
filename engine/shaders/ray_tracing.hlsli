#include "common.hlsli"

#define RT_EPSILON 1e-8

#define INTERSECTION_FLAG_NONE        0
#define INTERSECTION_FLAG_NORMAL      1 << 0
#define INTERSECTION_FLAG_TEXCOORD    1 << 1

static uint rt_top_nodes_offset;

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

TriangleIntersection ray_triangle_intersection(in TriInterInput input) {
    TriangleIntersection inter;
    inter.intersected = false;
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

    if (det < 0)
        // if the determinant is negative the triangle is backfacing
        return inter;

    if (abs(det) < RT_EPSILON)
        // if the determinant is 0, the ray misses the triangle
        return inter;
    
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

