pub mod management;

use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::ui::{
    Constraint, ContentFlow, CrossAlign, FlowAlign, Overflow, Position, Rect, RectUniformData, Sizing,
    UIEventHandlerC, UILayoutC, UILayoutCacheC,
};
use crate::ecs::component::{MeshRenderConfigC, SceneEventHandler, TransformC, UniformDataC, VertexMeshC};
use crate::event::WSIEvent;
use crate::module::scene::change_manager::ComponentChangesHandle;
use crate::module::scene::{EntityAccess, Scene, SceneObject};
use crate::module::ui::management::UIState;
use crate::module::EngineModule;
use crate::EngineContext;
use common::glm::{DVec3, Vec2, Vec3};
use common::scene::relation::Relation;
use common::types::Bool;
use common::types::HashSet;
use entity_data::{Archetype, EntityId, SystemAccess};
use smallvec::SmallVec;
use winit::window::Window;

pub struct UIRenderer {
    root_ui_entity: EntityId,
    root_element_size: Vec2,
    scale_factor: f32,
    root_element_size_dirty: bool,
    force_update_transforms: bool,
    ui_layout_changes: ComponentChangesHandle,
}

#[derive(Archetype)]
pub struct UIObject<S>
where
    S: UIState,
{
    relation: Relation,
    global_transform: GlobalTransformC,

    transform: TransformC,
    renderer: MeshRenderConfigC,
    uniforms: UniformDataC,
    mesh: VertexMeshC,

    scene_event_handler: SceneEventHandler,
    layout_cache: UILayoutCacheC,
    layout: UILayoutC,
    ui_event_handler: UIEventHandlerC,

    pub state: S,
}

impl<E: UIState> UIObject<E> {
    pub fn new_raw(layout: UILayoutC, state: E) -> Self {
        Self {
            relation: Default::default(),
            global_transform: Default::default(),
            transform: TransformC::new().with_use_parent_transform(false),
            renderer: Default::default(),
            uniforms: Default::default(),
            mesh: Default::default(),
            scene_event_handler: Default::default(),
            layout_cache: Default::default(),
            layout,
            ui_event_handler: Default::default(),
            state,
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

    pub fn with_scene_event_handler(mut self, handler: SceneEventHandler) -> Self {
        self.scene_event_handler = handler;
        self
    }

    pub fn disable_pointer_events(mut self) -> Self {
        self.ui_event_handler.enabled = false;
        self
    }

    pub fn add_event_handler(mut self, handler: UIEventHandlerC) -> Self {
        self.ui_event_handler
            .on_cursor_enter
            .extend(&handler.on_cursor_enter);
        self.ui_event_handler
            .on_cursor_leave
            .extend(&handler.on_cursor_leave);
        self.ui_event_handler
            .on_mouse_press
            .extend(&handler.on_mouse_press);
        self.ui_event_handler
            .on_mouse_release
            .extend(&handler.on_mouse_release);
        self.ui_event_handler.on_click.extend(&handler.on_click);
        self
    }
}

impl<E: UIState> SceneObject for UIObject<E> {
    fn request_update_on_addition() -> bool {
        true
    }
}

pub trait UIObjectEntityImpl<S: UIState> {
    fn state(&self) -> &S;
    fn state_mut(&mut self) -> &mut S;
}

impl<S: UIState> UIObjectEntityImpl<S> for EntityAccess<'_, UIObject<S>> {
    fn state(&self) -> &S {
        self.get::<S>()
    }

    fn state_mut(&mut self) -> &mut S {
        self.get_mut::<S>()
    }
}

#[derive(Debug)]
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
) -> SmallVec<[(EntityId, f32); 128]> {
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

    children_sizings
        .iter()
        .map(move |child| {
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
        .collect()
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
    pub fn new(ctx: &EngineContext, root_entity: EntityId) -> Self {
        let mut scene = ctx.module_mut::<Scene>();
        let ui_layout_changes = scene.change_manager_mut().register_component_flow::<UILayoutC>();

        let root_ui_entity = scene
            .add_object(Some(root_entity), UIObject::new_raw(UILayoutC::new(), ()))
            .unwrap();

        Self {
            root_ui_entity,
            root_element_size: Default::default(),
            root_element_size_dirty: false,
            scale_factor: 1.0,
            ui_layout_changes,
            force_update_transforms: false,
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
                continue;
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
            root_cache.clip_rect = Rect {
                min: Vec2::from_element(0.0),
                max: root_cache.final_size,
            }
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
            let parent_clip = &parent_cache.clip_rect;

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

                // Set new cross-flow size
                if child_new_cross_size != child_cache.final_size[cross_flow_axis] {
                    child_cache.final_size[cross_flow_axis] = child_new_cross_size;
                    dirty_elements.insert(*child);
                }

                // Collect info for final size calculation
                children_flow_sizings.push(ChildFlowSizingInfo {
                    entity: *child,
                    min: child_cache.final_min_size[flow_axis],
                    sizing: child_layout.sizing[flow_axis],
                    constraint: child_layout.constraints[flow_axis],
                    overflow: child_layout.overflow[flow_axis],
                });
            }

            // Calculate flow-size of each child
            let calculated_children_flow_sizes =
                flow_calculate_children_sizes(parent_size[flow_axis], &children_flow_sizings);

            let mut children_positioning_infos: SmallVec<[ChildPositioningInfo; 128]> =
                SmallVec::with_capacity(children_flow_sizings.len());

            // Set children flow sizes
            for (child, child_new_flow_size) in &calculated_children_flow_sizes {
                let child_layout = layout_comps.get(child).unwrap();
                let child_cache = layout_cache_comps.get_mut(child).unwrap();

                if *child_new_flow_size != child_cache.final_size[flow_axis] {
                    child_cache.final_size[flow_axis] = *child_new_flow_size;
                    dirty_elements.insert(*child);
                }

                // Collect info for position calculation
                children_positioning_infos.push(ChildPositioningInfo {
                    entity: *child,
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
            );

            // Calculate clipping regions
            for (child, child_final_flow_size) in &calculated_children_flow_sizes {
                let child_cache = layout_cache_comps.get_mut(child).unwrap();

                let mut local_clip_rect = Rect::default();
                local_clip_rect.min = child_cache.relative_position;
                local_clip_rect.max = local_clip_rect.min;
                local_clip_rect.max[cross_flow_axis] +=
                    child_cache.final_size[cross_flow_axis].min(parent_size[cross_flow_axis]);
                local_clip_rect.max[flow_axis] += child_final_flow_size.min(parent_size[flow_axis]);

                let global_clip_rect = Rect {
                    min: parent_cache.global_position + local_clip_rect.min,
                    max: parent_cache.global_position + local_clip_rect.max,
                };

                let new_clip_rect = parent_clip.intersection(&global_clip_rect);

                if child_cache.clip_rect != new_clip_rect {
                    child_cache.clip_rect = new_clip_rect;
                    dirty_elements.insert(*child);
                }
            }
        }
    }

    fn calculate_transforms(
        linear_tree: &[EntityId],
        scene: &mut Scene,
        root_size: &Vec2,
        dirty_elements: &mut HashSet<EntityId>,
        force_update_all: bool,
    ) {
        let root_size_inv = Vec2::from_element(1.0).component_div(root_size);

        for (i, node) in linear_tree.iter().enumerate() {
            if !force_update_all && !dirty_elements.contains(node) {
                continue;
            }

            let mut entry = scene.entry(node);

            let Some(layout) = entry.get_checked::<UILayoutC>() else {
                continue;
            };
            let cache = entry.get::<UILayoutCacheC>();
            let transform = entry.get::<TransformC>();

            let norm_pos = cache.global_position.component_mul(&root_size_inv);
            let norm_size = cache.final_size.component_mul(&root_size_inv);

            let new_scale = Vec3::new(norm_size.x, norm_size.y, 1.0);
            let new_position = DVec3::new(
                norm_pos.x as f64,
                1.0 - if layout.shader_inverted_y {
                    norm_pos.y
                } else {
                    norm_size.y + norm_pos.y
                } as f64,
                1000.0 - (i as f64),
            );

            if new_scale != transform.scale || new_position != transform.position {
                let cache = entry.get_mut::<UILayoutCacheC>();
                let rect = cache.clip_rect;
                cache.calculated_clip_rect = RectUniformData {
                    min: rect.min.component_mul(&root_size_inv),
                    max: rect.max.component_mul(&root_size_inv),
                };

                let transform = entry.get_mut::<TransformC>();
                transform.scale = new_scale;
                transform.position = new_position;

                // if parent transform changed, then all children transforms must be recalculated
                let relation = entry.get_mut::<Relation>();
                for child in &relation.children {
                    dirty_elements.insert(*child);
                }
            }
        }
    }

    /// Outputs objects that contain the specified point to the specified closure.
    /// If the closure returns `true` the traversal is stopped.
    pub fn traverse_at_point<F: FnMut(EntityAccess<()>) -> bool>(
        &self,
        point: &Vec2,
        scene: &mut Scene,
        mut output: F,
    ) {
        let linear_tree =
            common::scene::collect_relation_tree(&scene.storage_mut().access(), &self.root_ui_entity);

        // Iterate starting from children
        for node in linear_tree.iter().rev() {
            let entry = scene.entry(node);
            let Some(cache) = entry.get_checked::<UILayoutCacheC>() else {
                continue;
            };

            if !cache.clip_rect.contains_point(point) {
                continue;
            }
            if output(entry) {
                break;
            }
        }
    }
}

impl EngineModule for UIRenderer {
    fn on_update(&mut self, _: f64, ctx: &EngineContext) {
        let mut scene = ctx.module_mut::<Scene>();

        if self.root_element_size_dirty {
            let mut root = scene.entry(&self.root_ui_entity);
            let layout = root.get_mut::<UILayoutC>();

            layout.constraints[0] = Constraint::exact(self.root_element_size.x);
            layout.constraints[1] = Constraint::exact(self.root_element_size.y);

            self.root_element_size_dirty = false;
        }

        let mut dirty_elements: HashSet<_> = scene.change_manager_mut().take_new(self.ui_layout_changes);

        if !self.force_update_transforms && dirty_elements.is_empty() {
            return;
        }

        let storage = scene.storage_mut();
        let access = storage.access();
        let linear_tree = common::scene::collect_relation_tree(&access, &self.root_ui_entity);

        Self::calculate_final_minimum_sizes(&linear_tree, &access, &mut dirty_elements);
        Self::expand_sizes_and_set_positions(&linear_tree, &access, &mut dirty_elements);

        Self::calculate_transforms(
            &linear_tree,
            &mut scene,
            &self.root_element_size,
            &mut dirty_elements,
            self.force_update_transforms,
        );

        self.force_update_transforms = false;
    }

    fn on_wsi_event(&mut self, _: &Window, event: &WSIEvent, _: &EngineContext) {
        match event {
            WSIEvent::Resized(new_size) => {
                self.set_scale_factor(new_size.scale_factor());
                self.set_root_element_size(new_size.logical());
                self.force_update_transforms = true;
            }
            _ => {}
        }
    }
}
