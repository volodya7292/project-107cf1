use crate::rendering::textured_block_model::{PackedVertex, Vertex};
use crate::resource_mapping::ResourceMapping;
use base::overworld::accessor::ClusterNeighbourhoodAccessor;
use base::overworld::facing::Facing;
use base::overworld::light_state::LightLevel;
use base::overworld::liquid_state::LiquidState;
use base::overworld::occluder::Occluder;
use base::overworld::position::RelativeBlockPos;
use base::overworld::raw_cluster::RawCluster;
use base::overworld::raw_cluster::{BlockDataImpl, LightType};
use common::glm;
use common::glm::{I32Vec3, Vec2, Vec3};
use common::types::Bool;
use engine::module::main_renderer::vertex_mesh::{VAttributes, VertexMeshCreate};
use engine::module::main_renderer::VertexMesh;
use engine::vkw;
use std::sync::Arc;

#[derive(Default)]
pub struct ClusterMeshes {
    pub solid: VertexMesh<PackedVertex, ()>,
    pub transparent: VertexMesh<PackedVertex, ()>,
}

pub trait ClusterMeshBuilder {
    fn build_mesh(&self, device: &Arc<vkw::Device>, res_map: &ResourceMapping) -> ClusterMeshes;
}

/// Returns a corner and two sides corresponding to the specified vertex on the given block and facing.
#[inline]
fn calculate_ao(
    accessor: &ClusterNeighbourhoodAccessor,
    block_pos: &RelativeBlockPos,
    vertex_pos: &Vec3,
    facing: Facing,
) -> u8 {
    let registry = accessor.registry();
    let facing_dir = facing.direction();
    let facing_comp = facing_dir.iter().cloned().position(|v| v != 0).unwrap_or(0);

    let mut vertex_pos = *vertex_pos;
    vertex_pos[facing_comp] = 0.0;

    let mut center = Vec3::from_element(0.5);
    center[facing_comp] = 0.0;

    let side_dir: I32Vec3 = glm::convert_unchecked(glm::sign(&(vertex_pos - center)));

    let side1_comp = (facing_comp + 1) % 3;
    let side2_comp = (facing_comp + 2) % 3;

    let mut side1_dir = I32Vec3::default();
    side1_dir[side1_comp] = side_dir[side1_comp];

    let mut side2_dir = I32Vec3::default();
    side2_dir[side2_comp] = side_dir[side2_comp];

    let new_pos = block_pos.offset(facing_dir);
    let corner_pos = new_pos.offset(&side_dir);
    let side1_pos = new_pos.offset(&side1_dir);
    let side2_pos = new_pos.offset(&side2_dir);

    let corner = accessor.get_block(&corner_pos);
    let side1 = accessor.get_block(&side1_pos);
    let side2 = accessor.get_block(&side2_pos);

    let corner_occluder = corner.map_or(Occluder::EMPTY, |v| {
        registry.get_block(v.block_id()).unwrap().occluder()
    });
    let side1_occluder = side1.map_or(Occluder::EMPTY, |v| {
        registry.get_block(v.block_id()).unwrap().occluder()
    });
    let side2_occluder = side2.map_or(Occluder::EMPTY, |v| {
        registry.get_block(v.block_id()).unwrap().occluder()
    });

    (corner_occluder.is_empty() && side1_occluder.is_empty() && side2_occluder.is_empty()) as u8 * 255
}

#[inline]
fn calc_quad_lighting(
    accessor: &ClusterNeighbourhoodAccessor,
    block_pos: &RelativeBlockPos,
    quad: &[Vertex; 4],
    facing: Facing,
    light_type: LightType,
) -> [u16; 4] {
    const RELATIVE_SIDES: [[I32Vec3; 4]; 6] = [
        [
            *Facing::NegativeY.direction(),
            *Facing::NegativeZ.direction(),
            *Facing::PositiveY.direction(),
            *Facing::PositiveZ.direction(),
        ],
        [
            *Facing::NegativeY.direction(),
            *Facing::NegativeZ.direction(),
            *Facing::PositiveY.direction(),
            *Facing::PositiveZ.direction(),
        ],
        [
            *Facing::NegativeX.direction(),
            *Facing::NegativeZ.direction(),
            *Facing::PositiveX.direction(),
            *Facing::PositiveZ.direction(),
        ],
        [
            *Facing::NegativeX.direction(),
            *Facing::NegativeZ.direction(),
            *Facing::PositiveX.direction(),
            *Facing::PositiveZ.direction(),
        ],
        [
            *Facing::NegativeX.direction(),
            *Facing::NegativeY.direction(),
            *Facing::PositiveX.direction(),
            *Facing::PositiveY.direction(),
        ],
        [
            *Facing::NegativeX.direction(),
            *Facing::NegativeY.direction(),
            *Facing::PositiveX.direction(),
            *Facing::PositiveY.direction(),
        ],
    ];

    fn blend_lighting(base: LightLevel, mut neighbours: [LightLevel; 3]) -> Vec3 {
        for n in &mut neighbours {
            if n.is_zero() {
                *n = base;
            }
        }

        glm::convert::<_, Vec3>(
            base.components()
                + neighbours[0].components()
                + neighbours[1].components()
                + neighbours[2].components(),
        ) / (LightLevel::MAX_COMPONENT_VALUE as f32)
            * 0.25
    }

    let sides = &RELATIVE_SIDES[facing as usize];

    // FIXME: if block shape is not of full block, rel_pos = block_pos.
    let dir = facing.direction();
    let rel_pos = block_pos.offset(dir);

    let base = accessor
        .get_block(&rel_pos)
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let side0 = accessor
        .get_block(&rel_pos.offset(&sides[0]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let side1 = accessor
        .get_block(&rel_pos.offset(&sides[1]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let side2 = accessor
        .get_block(&rel_pos.offset(&sides[2]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let side3 = accessor
        .get_block(&rel_pos.offset(&sides[3]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));

    let corner01 = accessor
        .get_block(&rel_pos.offset(&sides[0]).offset(&sides[1]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let corner12 = accessor
        .get_block(&rel_pos.offset(&sides[1]).offset(&sides[2]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let corner23 = accessor
        .get_block(&rel_pos.offset(&sides[2]).offset(&sides[3]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));
    let corner30 = accessor
        .get_block(&rel_pos.offset(&sides[3]).offset(&sides[0]))
        .map_or(LightLevel::ZERO, |v| v.light_state_by(light_type));

    let lights = [
        blend_lighting(base, [side3, side0, corner30]),
        blend_lighting(base, [side2, side3, corner23]),
        blend_lighting(base, [side0, side1, corner01]),
        blend_lighting(base, [side1, side2, corner12]),
    ];

    const ADJACENT_COORD_IDS: [[usize; 2]; 3] = [[1, 2], [0, 2], [0, 1]];

    let facing_comp = dir.iamax();
    let [vi, vj] = ADJACENT_COORD_IDS[facing_comp];

    let mut result = [0_u16; 4];

    for (v_idx, v) in quad.iter().enumerate() {
        let ij = Vec2::new(v.position[vi], v.position[vj]);
        let inv_v = Vec2::from_element(1.0) - ij;
        let weights = [inv_v.x * inv_v.y, ij.x * inv_v.y, inv_v.x * ij.y, ij.x * ij.y];

        let lighting =
            weights[0] * lights[2] + weights[1] * lights[3] + weights[2] * lights[0] + weights[3] * lights[1];

        result[v_idx] = LightLevel::from_color(lighting).raw();
    }

    result
}

// `+ 1` in height is not needed
type LiquidHeightsCache =
    Box<[[[Option<f32>; RawCluster::SIZE + 1]; RawCluster::SIZE]; RawCluster::SIZE + 1]>;

/// Calculates liquid height at vertex in XZ-range 0..25
fn calc_liquid_height(
    accessor: &ClusterNeighbourhoodAccessor,
    cache: &mut LiquidHeightsCache,
    pos: &RelativeBlockPos,
) -> f32 {
    let registry = accessor.registry();
    let entry = &mut cache[pos.0.x as usize][pos.0.y as usize][pos.0.z as usize];

    // Use a cache to reduce the number of calculations by a factor of ~4
    *entry.get_or_insert_with(|| {
        let mut height_sum = 0_u32;
        let mut count = 0_u32;
        let mut level_bias_allowed = false;

        'outer: for i in -1..=0_i32 {
            for j in -1..=0_i32 {
                let rel_pos = pos.offset(&glm::vec3(i, 0, j));
                // let rel_padded_idx = aligned_block_index(&glm::convert_unchecked(rel_padded));
                let Some(rel_data) = accessor.get_block(&rel_pos) else {
                    continue;
                };

                let rel_liquid = rel_data.liquid_state();
                let rel_block = registry.get_block(rel_data.block_id()).unwrap();

                if !rel_block.can_pass_liquid() {
                    continue;
                }

                let top_liquid_exists = {
                    let top_rel_pos = rel_pos.offset(Facing::PositiveY.direction());
                    let top_level = accessor
                        .get_block(&top_rel_pos)
                        .map_or(0, |v| v.liquid_state().level());
                    top_level > 0
                };
                if rel_liquid.is_max() {
                    // All four liquid levels must be highest to ensure correct vertex joints
                    height_sum = LiquidState::MAX_LEVEL as u32;
                    count = 1;
                    level_bias_allowed = !top_liquid_exists;
                    break 'outer;
                }

                height_sum += rel_liquid.level() as u32;
                count += 1;
            }
        }

        // if there is no liquid above all four corners, lower current level slightly
        let top_factor = 1.0 - level_bias_allowed.into_f32() * 0.1;
        let factor = height_sum as f32 / (count * LiquidState::MAX_LEVEL as u32) as f32;

        top_factor * factor
    })
}

fn construct_liquid_quad(
    posf: &Vec3,
    material_id: u16,
    liquid_heights: &[f32; 4],
    facing: Facing,
    light_level: LightLevel,
    sky_light_level: LightLevel,
    vertices: &mut Vec<PackedVertex>,
) {
    const P000: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    const P001: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    const P100: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    const P101: Vec3 = Vec3::new(1.0, 0.0, 1.0);

    // let liq_p010 = Vec3::new(0.0, liquid_heights[0], 0.0);
    // let liq_p011 = Vec3::new(0.0, liquid_heights[1], 1.0);
    // let liq_p110 = Vec3::new(1.0, liquid_heights[2], 0.0);
    // let liq_p111 = Vec3::new(1.0, liquid_heights[3], 1.0);

    /*


    [
        // Z
        Quad::new([p110, p100, p010, p000]),
        Quad::new([p011, p001, p111, p101]),
        // Y
        Quad::new([p000, p100, p001, p101]),
        Quad::new([p011, p111, p010, p110]),
        // X
        Quad::new([p010, p000, p011, p001]),
        Quad::new([p111, p101, p110, p100]),
    ]

        // UV
        // Vec2::new(0.0, 0.0),
        // Vec2::new(0.0, 1.0),
        // Vec2::new(1.0, 0.0),
        // Vec2::new(1.0, 1.0),

     */

    let material_id = material_id as u32;
    let lighting = light_level.raw();
    let sky_lighting = sky_light_level.raw();
    let quad_vertices;
    let mut normal: Vec3 = glm::convert(*facing.direction());

    let quad_uv = [
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
    ];

    match facing {
        Facing::NegativeY => {
            quad_vertices = [P000, P100, P001, P101];
        }
        Facing::PositiveY => {
            quad_vertices = [
                Vec3::new(0.0, liquid_heights[1], 1.0),
                Vec3::new(1.0, liquid_heights[3], 1.0),
                Vec3::new(0.0, liquid_heights[0], 0.0),
                Vec3::new(1.0, liquid_heights[2], 0.0),
            ];
            normal =
                common::utils::calc_triangle_normal(&quad_vertices[0], &quad_vertices[1], &quad_vertices[2]);
        }
        Facing::NegativeX => {
            quad_vertices = [
                Vec3::new(0.0, liquid_heights[0], 0.0),
                P000,
                Vec3::new(0.0, liquid_heights[1], 1.0),
                P001,
            ];
        }
        Facing::PositiveX => {
            quad_vertices = [
                Vec3::new(1.0, liquid_heights[3], 1.0),
                P101,
                Vec3::new(1.0, liquid_heights[2], 0.0),
                P100,
            ];
        }
        Facing::NegativeZ => {
            quad_vertices = [
                Vec3::new(1.0, liquid_heights[2], 0.0),
                P100,
                Vec3::new(0.0, liquid_heights[0], 0.0),
                P000,
            ];
        }
        Facing::PositiveZ => {
            quad_vertices = [
                Vec3::new(0.0, liquid_heights[1], 1.0),
                P001,
                Vec3::new(1.0, liquid_heights[3], 1.0),
                P101,
            ];
        }
    };

    vertices.extend(
        quad_vertices
            .into_iter()
            .zip(quad_uv.into_iter())
            .map(|(pos, uv)| {
                Vertex {
                    position: posf + pos,
                    normal,
                    tex_uv: uv,
                    ao: u8::MAX,
                    lighting,
                    sky_lighting,
                    material_id,
                }
                .pack()
            }),
    );
}

fn gen_block_vertices(
    accessor: &ClusterNeighbourhoodAccessor,
    res_map: &ResourceMapping,
    pos: &RelativeBlockPos,
    liquid_cache: &mut LiquidHeightsCache,
    vertices: &mut Vec<PackedVertex>,
    vertices_translucent: &mut Vec<PackedVertex>,
) {
    let registry = accessor.registry();

    let state = accessor.get_block(pos).unwrap();
    let block = registry.get_block(state.block_id()).unwrap();

    let posf: Vec3 = glm::convert(pos.0);
    let model = res_map.textured_model_for_block(state.block_id());

    let contains_liquid = state.liquid_state().level() > 0;
    let mut liquid_material_id = u16::MAX;
    let mut liquid_heights = [0_f32; 4];

    // Calculate liquid heights if it is present
    if contains_liquid {
        liquid_material_id = res_map.material_for_liquid(state.liquid_state().liquid_id());
        for x in 0..2 {
            for z in 0..2 {
                let rel = pos.offset(&glm::vec3(x, 0, z));
                let height = calc_liquid_height(accessor, liquid_cache, &rel);
                liquid_heights[(x * 2 + z) as usize] = height;
            }
        }
    } else if block.is_model_invisible() {
        // Nothing to render
        return;
    }

    // Generate inner faces
    if !block.is_model_invisible() && !model.get_inner_quads().is_empty() {
        // TODO: REMOVE: For inner quads use the light level of the current block
        // let aligned_pos = pos.add_scalar(1);
        // let index = aligned_block_index(&glm::convert_unchecked(aligned_pos));
        // let light_level = intrinsics[index].light_level;

        for quad in model.get_inner_quads() {
            let mut quad_vertices = quad.vertices;
            let normal = common::utils::calc_triangle_normal(
                &quad_vertices[0].position,
                &quad_vertices[1].position,
                &quad_vertices[2].position,
            );

            let closest_facing = Facing::from_normal_closest(&normal);
            let regular_lighting =
                calc_quad_lighting(accessor, pos, &quad_vertices, closest_facing, LightType::Regular);
            let sky_lighting =
                calc_quad_lighting(accessor, pos, &quad_vertices, closest_facing, LightType::Sky);

            for ((vert, regular), sky) in quad_vertices.iter_mut().zip(regular_lighting).zip(sky_lighting) {
                vert.lighting = regular;
                vert.sky_lighting = sky;
            }

            let vertices_vec = if quad.transparent {
                &mut *vertices_translucent
            } else {
                &mut *vertices
            };

            for mut v in quad_vertices {
                v.position += posf;
                v.normal = normal;

                vertices_vec.push(v.pack());
            }
        }
    }

    // Generate side faces
    for i in 0..6 {
        let facing = Facing::from_u8(i as u8);
        let rel_pos = pos.offset(facing.direction());
        let rel_cell = accessor.get_block(&rel_pos);

        // Render liquid if present
        if let Some(rel_cell) = rel_cell {
            let rel_block = registry.get_block(rel_cell.block_id()).unwrap();
            let rel_occludes = rel_block.occluder().occludes_side(facing.mirror());

            // Render liquid quad if liquid is present
            if contains_liquid
                && (rel_cell.liquid_state().level() == 0
                    || (facing == Facing::NegativeY && !rel_cell.liquid_state().is_max())
                    || (facing == Facing::PositiveY && !state.liquid_state().is_max()))
            {
                // Add respective liquid quad
                construct_liquid_quad(
                    &posf,
                    liquid_material_id,
                    &liquid_heights,
                    facing,
                    state.light_state(),
                    state.sky_light_state(),
                    vertices_translucent,
                );
            }

            // Do not emit face if this side is fully occluded by neighbouring block
            if rel_occludes {
                continue;
            }

            if block.is_model_invisible() {
                // The model is invisible, skip further rendering
                continue;
            }

            // Do not emit face if certain conditions are met
            if model.merge_enabled() {
                if state.block_id() == rel_cell.block_id() && model.side_shapes_equality()[facing.axis_idx()]
                {
                    // Side faces are of the same shape
                    continue;
                } else {
                    let rel_model = res_map.textured_model_for_block(rel_cell.block_id());
                    if model
                        .first_side_quad_vsorted(facing)
                        .cmp_ordered(rel_model.first_side_quad_vsorted(facing.mirror()))
                    {
                        // `model` and `rel_model` have side quad of the same shape
                        continue;
                    }
                }
            }
        }

        if block.is_model_invisible() {
            // do not render side faces
            continue;
        }

        // Render model side faces
        for quad in model.get_quads_by_facing(facing) {
            let mut quad_vertices = quad.vertices;

            let normal = common::utils::calc_triangle_normal(
                &quad_vertices[0].position,
                &quad_vertices[1].position,
                &quad_vertices[2].position,
            );

            let regular_lighting =
                calc_quad_lighting(accessor, pos, &quad_vertices, facing, LightType::Regular);
            let sky_lighting = calc_quad_lighting(accessor, pos, &quad_vertices, facing, LightType::Sky);

            for ((v, regular_light), sky_light) in
                quad_vertices.iter_mut().zip(regular_lighting).zip(sky_lighting)
            {
                let ao = calculate_ao(accessor, pos, &v.position, facing);
                v.position += posf;
                v.normal = normal;
                v.ao = ao;
                v.lighting = regular_light;
                v.sky_lighting = sky_light;
            }

            if quad_vertices[1].ao != quad_vertices[2].ao
                || quad_vertices[1].lighting != quad_vertices[2].lighting
            {
                let vc = quad_vertices;
                quad_vertices[1] = vc[0];
                quad_vertices[3] = vc[1];
                quad_vertices[0] = vc[2];
                quad_vertices[2] = vc[3];
            }

            let vertices_vec = if quad.transparent {
                &mut *vertices_translucent
            } else {
                &mut *vertices
            };

            vertices_vec.extend(quad_vertices.map(|v| v.pack()));
        }
    }
}

impl ClusterMeshBuilder for ClusterNeighbourhoodAccessor {
    fn build_mesh(&self, device: &Arc<vkw::Device>, res_map: &ResourceMapping) -> ClusterMeshes {
        let mut vertices = Vec::<PackedVertex>::with_capacity(RawCluster::VOLUME * 8);
        let mut vertices_translucent = Vec::<PackedVertex>::with_capacity(RawCluster::VOLUME * 8);
        let mut liquid_cache = LiquidHeightsCache::default();

        for x in 0..RawCluster::SIZE {
            for y in 0..RawCluster::SIZE {
                for z in 0..RawCluster::SIZE {
                    let pos = I32Vec3::new(x as i32, y as i32, z as i32);
                    gen_block_vertices(
                        self,
                        res_map,
                        &RelativeBlockPos(pos),
                        &mut liquid_cache,
                        &mut vertices,
                        &mut vertices_translucent,
                    );
                }
            }
        }

        let mut indices = vec![0; vertices.len() / 4 * 6];
        let mut indices_translucent = vec![0; vertices_translucent.len() / 4 * 6];

        let map_quad_ids = |quad_idx, chunk: &mut [u32]| {
            let ind = (quad_idx * 4) as u32;
            chunk[0] = ind;
            chunk[1] = ind + 2;
            chunk[2] = ind + 1;
            chunk[3] = ind + 2;
            chunk[4] = ind + 3;
            chunk[5] = ind + 1;
        };

        for (i, chunk) in indices.chunks_exact_mut(6).enumerate() {
            map_quad_ids(i, chunk);
        }
        for (i, chunk) in indices_translucent.chunks_exact_mut(6).enumerate() {
            map_quad_ids(i, chunk);
        }

        let meshes = ClusterMeshes {
            solid: device
                .create_vertex_mesh(VAttributes::Slice(&vertices), Some(&indices))
                .unwrap(),
            transparent: device
                .create_vertex_mesh(
                    VAttributes::Slice(&vertices_translucent),
                    Some(&indices_translucent),
                )
                .unwrap(),
        };

        meshes
    }
}
