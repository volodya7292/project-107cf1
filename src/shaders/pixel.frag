#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragTexColor;
layout(location = 2) in vec4 fragCoord;
layout(location = 3) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

// trilinear method

void main() {
    //vec4 tex_x = texture(texSampler, coord_one.yz);
    //vec4 tex_y = texture(texSampler, coord_one.xz);
    //vec4 tex_z = texture(texSampler, coord_one.xy);

    //vec4 m = (tex_x + tex_y + tex_z) / 3.0;



    vec3 blend_weights = abs(fragNormal);
    blend_weights = (blend_weights - 0.2) * 7;
    blend_weights = max(blend_weights, 0);
    blend_weights /= (blend_weights.x + blend_weights.y + blend_weights.z);

    vec2 coord_x = fragCoord.yz;
    vec2 coord_y = fragCoord.zx;
    vec2 coord_z = fragCoord.xy;

    vec4 col_x = texture(texSampler, coord_x);
    vec4 col_y = texture(texSampler, coord_y);
    vec4 col_z = texture(texSampler, coord_z);

    vec4 blended_color = col_x.xyzw * blend_weights.x +
                        col_y.xyzw * blend_weights.y +
                        col_z.xyzw * blend_weights.z;

    // // Determine the blend weights for the 3 planar projections.
    // // N_orig is the vertex-interpolated normal vector.
    // float3 blend_weights = abs( N_orig.xyz );   // Tighten up the blending zone:
    // blend_weights = (blend_weights - 0.2) * 7;
    // blend_weights = max(blend_weights, 0);      // Force weights to sum to 1.0 (very important!)
    // blend_weights /= (blend_weights.x + blend_weights.y + blend_weights.z ).xxx; 
    // // Now determine a color value and bump vector for each of the 3
    // // projections, blend them, and store blended results in these two
    // // vectors:
    // float4 blended_color; // .w hold spec value
    // float3 blended_bump_vec;
    // {
    //     // Compute the UV coords for each of the 3 planar projections.
    //     // tex_scale (default ~ 1.0) determines how big the textures appear.
    //     float2 coord1 = v2f.wsCoord.yz * tex_scale;
    //     float2 coord2 = v2f.wsCoord.zx * tex_scale;
    //     float2 coord3 = v2f.wsCoord.xy * tex_scale;
    //     // This is where you would apply conditional displacement mapping.
    //     //if (blend_weights.x > 0) coord1 = . . .
    //     //if (blend_weights.y > 0) coord2 = . . .
    //     //if (blend_weights.z > 0) coord3 = . . .
    //     // Sample color maps for each projection, at those UV coords.
    //     float4 col1 = colorTex1.Sample(coord1);
    //     float4 col2 = colorTex2.Sample(coord2);
    //     float4 col3 = colorTex3.Sample(coord3);
    //     // Sample bump maps too, and generate bump vectors.
    //     // (Note: this uses an oversimplified tangent basis.)
    //     float2 bumpFetch1 = bumpTex1.Sample(coord1).xy - 0.5;
    //     float2 bumpFetch2 = bumpTex2.Sample(coord2).xy - 0.5;
    //     float2 bumpFetch3 = bumpTex3.Sample(coord3).xy - 0.5;
    //     float3 bump1 = float3(0, bumpFetch1.x, bumpFetch1.y);
    //     float3 bump2 = float3(bumpFetch2.y, 0, bumpFetch2.x);
    //     float3 bump3 = float3(bumpFetch3.x, bumpFetch3.y, 0);
    //     // Finally, blend the results of the 3 planar projections.
    //     blended_color = col1.xyzw * blend_weights.xxxx +
    //                     col2.xyzw * blend_weights.yyyy +
    //                     col3.xyzw * blend_weights.zzzz;
    //     blended_bump_vec = bump1.xyz * blend_weights.xxx +
    //                     bump2.xyz * blend_weights.yyy +
    //                     bump3.xyz * blend_weights.zzz;
    // }
    // // Apply bump vector to vertex-interpolated normal vector.
    // float3 N_for_lighting = normalize(N_orig + blended_bump);



    outColor = vec4(blended_color.xyz, 1);
}
