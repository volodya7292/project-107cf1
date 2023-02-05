pub mod element;
pub mod management;

use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::ui::{
    Constraint, ContentFlow, CrossAlign, FlowAlign, Overflow, Position, Sizing, UILayoutC, UILayoutCacheC,
};
use crate::ecs::component::{EventHandlerC, MeshRenderConfigC, TransformC, VertexMeshC};
use crate::ecs::SceneAccess;
use crate::renderer::module::RendererModule;
use crate::renderer::{DirtyComponents, Renderer, RendererContext, SceneObject};
use base::scene::relation::Relation;
use base::utils::{Bool, HashSet};
use entity_data::{Archetype, EntityId, SystemAccess};
pub use management::UIState;
use nalgebra_glm::{DVec3, Vec2, Vec3};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::{CmdList, Device};

pub struct UIRenderer {
    device: Arc<Device>,
    root_ui_entity: EntityId,
    root_element_size: Vec2,
    root_element_size_dirty: bool,
    scale_factor: f32,
}

pub struct DefaultStyling {}

#[derive(Archetype)]
pub struct UIObject<S>
where
    S: UIState,
{
    relation: Relation,
    global_transform: GlobalTransformC,

    transform: TransformC,
    renderer: MeshRenderConfigC,
    mesh: VertexMeshC,

    event_handler: EventHandlerC,
    layout_cache: UILayoutCacheC,
    layout: UILayoutC,

    element: S,
}

impl<E: UIState> UIObject<E> {
    pub fn new_raw(layout: UILayoutC, element: E) -> Self {
        Self {
            relation: Default::default(),
            global_transform: Default::default(),
            transform: TransformC::new().with_use_parent_transform(false),
            renderer: Default::default(),
            mesh: Default::default(),
            event_handler: Default::default(),
            layout_cache: Default::default(),
            layout,
            element,
        }
    }

    pub fn with_renderer(mut self, renderer: MeshRenderConfigC) -> Self {
        self.renderer = renderer;
        self
    }

    pub fn with_mesh(mut self, mesh: VertexMeshC) -> Self {
        self.mesh = mesh;
        self
    }

    pub fn with_event_handler(mut self, handler: EventHandlerC) -> Self {
        self.event_handler = handler;
        self
    }

    pub(crate) fn with_layout_cache(mut self, layout_cache: UILayoutCacheC) -> Self {
        self.layout_cache = layout_cache;
        self
    }
}

impl<E: UIState> SceneObject for UIObject<E> {}

struct ChildFlowSizingInfo {
    entity: EntityId,
    min: f32,
    sizing: Sizing,
    constraint: Constraint,
    overflow: Overflow,
}

/// Calculates final size for each child in a flow. Outputs pairs of [EntityId] and corresponding size.
fn flow_calculate_children_sizes(
    parent_size: f32,
    children_sizings: &[ChildFlowSizingInfo],
) -> impl Iterator<Item = (EntityId, f32)> + '_ {
    let mut sum_preferred = 0.0;
    let mut sum_grow_factor = 0.0;
    let mut max_space_for_preferred = parent_size;

    for child in children_sizings {
        match child.sizing {
            Sizing::Grow(factor) => sum_grow_factor += factor,
            _ => {}
        }

        if let Sizing::Preferred(size) = child.sizing {
            sum_preferred += child.constraint.clamp(size);
        } else {
            max_space_for_preferred = (max_space_for_preferred - child.min).max(0.0);
        }
    }

    max_space_for_preferred = max_space_for_preferred.min(sum_preferred);
    sum_grow_factor = sum_grow_factor.max(1.0);

    let max_space_for_grow = (parent_size - max_space_for_preferred).max(0.0);
    let mut preferred_space_left = max_space_for_preferred;
    let mut grow_space_left = max_space_for_grow;

    children_sizings.iter().map(move |child| {
        let final_size = match child.sizing {
            Sizing::Preferred(size) => {
                let min = child.min * (child.overflow == Overflow::Visible).into_f32();
                let proportion = size / sum_preferred;

                let allocated =
                    (max_space_for_preferred * proportion).clamp(min, preferred_space_left.max(min));
                let allocated_constrained = child.constraint.clamp(allocated);

                preferred_space_left -= allocated_constrained.min(preferred_space_left);
                allocated_constrained
            }
            Sizing::Grow(factor) => {
                let min = child.min * (child.overflow == Overflow::Visible).into_f32();
                let proportion = factor / sum_grow_factor;

                let allocated = (max_space_for_grow * proportion).clamp(min, grow_space_left.max(min));
                let allocated_constrained = child.constraint.clamp(allocated);

                grow_space_left -= allocated_constrained.min(grow_space_left);
                allocated_constrained
            }
            Sizing::FitContent => child.min,
        };
        (child.entity, final_size)
    })
}

struct ChildPositioningInfo {
    entity: EntityId,
    cross_align: CrossAlign,
    size: Vec2,
}

/// Calculates relative position for each child in a flow.
fn flow_calculate_children_positions<F: FnMut(EntityId, Vec2)>(
    parent_size: Vec2,
    flow: ContentFlow,
    flow_align: FlowAlign,
    children_sizes: &[ChildPositioningInfo],
    mut output: F,
) {
    let flow_axis = flow.axis();
    let cross_flow_axis = flow.cross_axis();
    let flow_size_sum: f32 = children_sizes.iter().map(|child| child.size[flow_axis]).sum();

    let mut curr_flow_pos = match flow_align {
        FlowAlign::Start => 0.0,
        FlowAlign::Center => (parent_size[flow_axis] - flow_size_sum) * 0.5,
        FlowAlign::End => parent_size[flow_axis] - flow_size_sum,
    };

    for child in children_sizes {
        let cross_flow_pos = match child.cross_align {
            CrossAlign::Start => 0.0,
            CrossAlign::Center => (parent_size[cross_flow_axis] - child.size[cross_flow_axis]) * 0.5,
            CrossAlign::End => parent_size[cross_flow_axis] - child.size[cross_flow_axis],
        };

        let mut pos = Vec2::default();
        pos[flow_axis] = curr_flow_pos;
        pos[cross_flow_axis] = cross_flow_pos;

        output(child.entity, pos);

        curr_flow_pos += child.size[flow_axis];
    }
}

impl UIRenderer {
    pub fn new(renderer: &mut Renderer) -> Self {
        let root_ui_entity = renderer
            .add_object(
                None,
                UIObject::new_raw(UILayoutC::new(), ()).with_layout_cache(UILayoutCacheC {
                    final_min_size: Default::default(),
                    final_size: Default::default(),
                    relative_position: Default::default(),
                    global_position: Default::default(),
                }),
            )
            .unwrap();

        Self {
            device: Arc::clone(&renderer.device),
            root_ui_entity,
            root_element_size: Default::default(),
            root_element_size_dirty: false,
            scale_factor: 1.0,
        }
    }

    pub fn root_ui_entity(&self) -> &EntityId {
        &self.root_ui_entity
    }

    fn set_root_element_size(&mut self, logical_size: Vec2) {
        self.root_element_size = logical_size;
        self.root_element_size_dirty = true;
    }

    fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = scale_factor;
        self.root_element_size_dirty = true;
    }

    /// Calculates minimum sizes for each element starting from children (bottom of the tree).
    fn calculate_final_minimum_sizes(
        linear_tree: &[EntityId],
        access: &SystemAccess,
        dirty_elements: &mut HashSet<EntityId>,
    ) {
        let relation_comps = access.component::<Relation>();
        let layout_comps = access.component::<UILayoutC>();
        let mut layout_cache_comps = access.component_mut::<UILayoutCacheC>();

        const ROOT_IDX: usize = 0;

        // Start from children (bottom of the tree)
        for (i, node) in linear_tree.iter().enumerate().rev() {
            if !dirty_elements.contains(node) {
                return;
            }

            let relation = relation_comps.get(node).unwrap();
            let layout = layout_comps.get(node).unwrap();
            let flow_axis = layout.content_flow.axis();
            let cross_flow_axis = layout.content_flow.cross_axis();
            let mut new_min_size = Vec2::from_element(0.0);

            // Calculate self min size using children min sizes
            for child in &relation.children {
                let Some(child_layout) = layout_comps.get(child) else {
                    continue;
                };
                if &child_layout.position != &Position::Auto {
                    continue;
                }
                let child_cache = layout_cache_comps.get(child).unwrap();

                let flow_size = &mut new_min_size[flow_axis];
                *flow_size += child_cache.final_min_size[flow_axis];

                let min_cross_size = &mut new_min_size[cross_flow_axis];
                *min_cross_size = min_cross_size.max(child_cache.final_min_size[cross_flow_axis]);
            }

            new_min_size += layout.padding.size();

            // Apply constraints to minimum size
            for (min_size, constraint) in new_min_size.iter_mut().zip(layout.constraints.iter()) {
                *min_size = constraint.clamp(*min_size);
            }

            let cache = layout_cache_comps.get_mut(node).unwrap();

            if cache.final_min_size == new_min_size {
                continue;
            }
            cache.final_min_size = new_min_size;

            if i != ROOT_IDX {
                // Make parent dirty
                dirty_elements.insert(relation.parent);
            }
        }
    }

    /// For each element calculates expanded size and its position (starting from the top of the tree).
    fn expand_sizes_and_set_positions(
        linear_tree: &[EntityId],
        access: &SystemAccess,
        dirty_elements: &mut HashSet<EntityId>,
    ) {
        let relation_comps = access.component::<Relation>();
        let layout_comps = access.component::<UILayoutC>();
        let mut layout_cache_comps = access.component_mut::<UILayoutCacheC>();

        // Set final size for the root element
        {
            let root = linear_tree[0];
            let root_cache = layout_cache_comps.get_mut(&root).unwrap();
            root_cache.final_size = root_cache.final_min_size;
        }

        // Start from top of the tree
        for node in linear_tree {
            let Some(relation) = relation_comps.get(node) else {
                continue;
            };

            let contains_dirty_children = relation
                .children
                .iter()
                .any(|child| dirty_elements.contains(child));

            if !dirty_elements.contains(node) || !contains_dirty_children {
                continue;
            }

            let parent_layout = layout_comps.get(node).unwrap();
            let parent_cache = *layout_cache_comps.get(node).unwrap();
            let flow_axis = parent_layout.content_flow.axis();
            let cross_flow_axis = parent_layout.content_flow.cross_axis();
            let parent_size = parent_cache.final_size;

            let mut children_flow_sizings: SmallVec<[ChildFlowSizingInfo; 128]> =
                SmallVec::with_capacity(relation.children.len());

            for child in &relation.children {
                let Some(child_layout) = layout_comps.get(child) else {
                    continue;
                };
                let child_cache = layout_cache_comps.get_mut(child).unwrap();

                // Calculate cross-flow-size for `child`
                let child_new_cross_size = child_layout.constraints[cross_flow_axis].clamp(
                    match child_layout.sizing[cross_flow_axis] {
                        Sizing::Preferred(size) => size.clamp(
                            child_cache.final_min_size[cross_flow_axis],
                            parent_size[cross_flow_axis],
                        ),
                        Sizing::FitContent => child_cache.final_min_size[cross_flow_axis],
                        Sizing::Grow(factor) => parent_size[cross_flow_axis] * factor.min(1.0),
                    },
                );
                let child_curr_cross_size = &mut child_cache.final_size[cross_flow_axis];

                if child_new_cross_size != *child_curr_cross_size {
                    *child_curr_cross_size = child_new_cross_size;
                    dirty_elements.insert(*child);
                }

                // Collect info for final size calculation
                children_flow_sizings.push(ChildFlowSizingInfo {
                    entity: *child,
                    min: child_cache.final_min_size[flow_axis],
                    sizing: child_layout.sizing[flow_axis],
                    constraint: child_layout.constraints[flow_axis],
                    overflow: child_layout.overflow,
                });
            }

            // Calculate flow-size of each child
            let calculated_children_sizes =
                flow_calculate_children_sizes(parent_size[flow_axis], &children_flow_sizings);

            let mut children_positioning_infos: SmallVec<[ChildPositioningInfo; 128]> =
                SmallVec::with_capacity(children_flow_sizings.len());

            for (child, child_new_flow_size) in calculated_children_sizes {
                let child_layout = layout_comps.get(&child).unwrap();
                let child_cache = layout_cache_comps.get_mut(&child).unwrap();
                let child_curr_flow_size = &mut child_cache.final_size[flow_axis];

                if child_new_flow_size != *child_curr_flow_size {
                    *child_curr_flow_size = child_new_flow_size;
                    dirty_elements.insert(child);
                }

                // Collect info for position calculation
                children_positioning_infos.push(ChildPositioningInfo {
                    entity: child,
                    cross_align: child_layout.align,
                    size: child_cache.final_size,
                });
            }

            // Calculate position of each child
            flow_calculate_children_positions(
                parent_size,
                parent_layout.content_flow,
                parent_layout.flow_align,
                &children_positioning_infos,
                |child, position| {
                    let child_cache = layout_cache_comps.get_mut(&child).unwrap();
                    let new_global_pos = parent_cache.global_position + position;

                    if child_cache.global_position != new_global_pos {
                        child_cache.global_position = new_global_pos;
                        dirty_elements.insert(child);
                    }

                    child_cache.relative_position = position;
                },
            )
        }
    }

    fn calculate_transforms(
        linear_tree: &[EntityId],
        scene: &mut SceneAccess<()>,
        root_size: &Vec2,
        dirty_elements: &mut HashSet<EntityId>,
    ) {
        let access = scene.storage.access();
        let mut dirty_comps = scene.dirty_components.borrow_mut();

        let mut transform_comps = access.component_mut::<TransformC>();
        let relation_comps = access.component::<Relation>();
        let layout_comps = access.component::<UILayoutC>();
        let layout_cache_comps = access.component_mut::<UILayoutCacheC>();

        let root_size_inv = Vec2::from_element(1.0).component_div(root_size);

        for (i, node) in linear_tree.iter().enumerate() {
            if !dirty_elements.contains(node) {
                continue;
            }

            let Some(transform) = transform_comps.get_mut(node) else {
                continue;
            };
            let layout = layout_comps.get(node).unwrap();
            let cache = layout_cache_comps.get(node).unwrap();

            let norm_pos = cache.global_position.component_mul(&root_size_inv);
            let norm_size = cache.final_size.component_mul(&root_size_inv);

            let new_scale = Vec3::new(norm_size.x, norm_size.y, 1.0);
            let new_position = DVec3::new(
                norm_pos.x as f64,
                1.0 - if layout.shader_inverted_y {
                    norm_pos.y
                } else {
                    norm_size.y - norm_pos.y
                } as f64,
                i as f64,
            );

            if new_scale != transform.scale || new_position != transform.position {
                transform.scale = new_scale;
                transform.position = new_position;

                // if parent transform changed, then all children transforms must be recalculated
                let relation = relation_comps.get(node).unwrap();
                for child in &relation.children {
                    dirty_elements.insert(*child);
                }
            }

            dirty_comps.add::<TransformC>(node);
        }
    }

    fn update_hierarchy(&mut self, scene: &mut SceneAccess<()>) {
        let mut dirty_elements = scene.dirty_components.borrow_mut().take_changes::<UILayoutC>();
        if dirty_elements.is_empty() {
            return;
        }

        let access = scene.storage.access();

        let linear_tree = {
            let layout_comps = access.component::<UILayoutC>();
            let mut linear_tree = Vec::with_capacity(layout_comps.count_entities());
            linear_tree.push(self.root_ui_entity);
            base::scene::collect_children_recursively(&access, &self.root_ui_entity, &mut linear_tree);
            linear_tree
        };

        Self::calculate_final_minimum_sizes(&linear_tree, &access, &mut dirty_elements);

        Self::expand_sizes_and_set_positions(&linear_tree, &access, &mut dirty_elements);

        Self::calculate_transforms(&linear_tree, scene, &self.root_element_size, &mut dirty_elements);
    }
}

impl RendererModule for UIRenderer {
    fn on_update(&mut self, mut scene: SceneAccess<()>) -> Option<Arc<Mutex<CmdList>>> {
        if self.root_element_size_dirty {
            let mut root = scene.entry_mut(&self.root_ui_entity).unwrap();
            let layout = root.get_mut::<UILayoutC>();

            layout.constraints[0] = Constraint::exact(self.root_element_size.x);
            layout.constraints[1] = Constraint::exact(self.root_element_size.y);

            self.root_element_size_dirty = false;
        }

        self.update_hierarchy(&mut scene);

        None
    }

    fn on_resize(&mut self, physical_size: (u32, u32), scale_factor: f64) {
        let scale_factor = scale_factor as f32;

        self.set_scale_factor(scale_factor);
        self.set_root_element_size(Vec2::new(physical_size.0 as f32, physical_size.1 as f32) / scale_factor);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
