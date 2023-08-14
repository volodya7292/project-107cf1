pub mod color;
pub mod reactive;

use crate::ecs::component::internal::HierarchyCacheC;
use crate::ecs::component::ui::{
    ClipRect, Constraint, ContentFlow, CrossAlign, FlowAlign, Overflow, Padding, Position, RectUniformData,
    Sizing, UIEventHandlerC, UILayoutC, UILayoutCacheC, Visibility,
};
use crate::ecs::component::{MeshRenderConfigC, SceneEventHandler, TransformC, UniformDataC, VertexMeshC};
use crate::event::WSIEvent;
use crate::module::scene::change_manager::ComponentChangesHandle;
use crate::module::scene::{EntityAccess, ObjectEntityId, Scene, SceneObject};
use crate::module::EngineModule;
use crate::EngineContext;
use common::glm;
use common::glm::{DVec3, Vec2, Vec3};
use common::scene::relation::Relation;
use common::types::HashSet;
use entity_data::{Archetype, EntityId};
use smallvec::SmallVec;
use std::sync::Arc;
use winit::window::Window;

pub struct UIRenderer {
    root_ui_entity: ObjectEntityId<StatelessUIObject>,
    root_element_size: Vec2,
    scale_factor: f32,
    root_element_size_dirty: bool,
    force_update_transforms: bool,
    ui_layout_changes: ComponentChangesHandle,
    relation_changes: ComponentChangesHandle,
}

pub trait UIState: Send + Sync + 'static {
    fn on_update(_entity: &EntityId, _ctx: &EngineContext, _dt: f64) {}
}

impl UIState for () {}

#[derive(Archetype)]
pub struct UIObject<S>
where
    S: UIState,
{
    relation: Relation,
    h_cache: HierarchyCacheC,

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
            h_cache: Default::default(),
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
}

impl<E: UIState> SceneObject for UIObject<E> {
    fn request_update_on_addition() -> bool {
        true
    }

    fn on_update(entity: &EntityId, ctx: &EngineContext, dt: f64) {
        E::on_update(entity, ctx, dt);
    }
}

pub type StatelessUIObject = UIObject<()>;

pub trait UIObjectEntityImpl<S: UIState> {
    fn state(&self) -> &S;
    fn state_mut(&mut self) -> &mut S;
    fn layout(&self) -> &UILayoutC;
    fn layout_mut(&mut self) -> &mut UILayoutC;
    fn event_handler_mut(&mut self) -> &mut UIEventHandlerC;
    fn layout_cache(&self) -> &UILayoutCacheC;
}

impl<S: UIState> UIObjectEntityImpl<S> for EntityAccess<'_, UIObject<S>> {
    fn state(&self) -> &S {
        self.get::<S>()
    }

    fn state_mut(&mut self) -> &mut S {
        self.get_mut::<S>()
    }

    fn layout(&self) -> &UILayoutC {
        self.get::<UILayoutC>()
    }

    fn layout_mut(&mut self) -> &mut UILayoutC {
        self.get_mut::<UILayoutC>()
    }

    fn event_handler_mut(&mut self) -> &mut UIEventHandlerC {
        self.get_mut::<UIEventHandlerC>()
    }

    fn layout_cache(&self) -> &UILayoutCacheC {
        self.get::<UILayoutCacheC>()
    }
}

#[derive(Debug)]
struct ChildFlowSizingInfo {
    entity: EntityId,
    min_flow_size: f32,
    max_flow_size: f32,
    relatively_positioned: bool,
    sizing: Sizing,
    overflow: Overflow,
}

/// Calculates final size for each child in a flow. Outputs pairs of [EntityId] and corresponding size.
fn flow_calculate_children_sizes(
    parent_size: Vec2,
    flow_axis: usize,
    children_sizings: &[ChildFlowSizingInfo],
) -> SmallVec<[(EntityId, f32); 128]> {
    let flow_parent_size = parent_size[flow_axis];

    let mut result: SmallVec<[(EntityId, f32); 128]> =
        children_sizings.iter().map(|info| (info.entity, 0.0)).collect();

    let mut sum_fit_content = 0.0;
    let mut sum_grow_factor = 0.0;
    let mut min_sum_preferred = 0.0;
    let mut min_sum_grow = 0.0;

    for child in children_sizings {
        if child.relatively_positioned {
            continue;
        }
        match child.sizing {
            Sizing::Preferred(_) => {
                min_sum_preferred += child.min_flow_size;
            }
            Sizing::Grow(factor) => {
                sum_grow_factor += factor;
                min_sum_grow += child.min_flow_size;
            }
            Sizing::FitContent | Sizing::ParentBased(_) => {
                sum_fit_content += child.min_flow_size;
            }
        }
    }
    sum_grow_factor = sum_grow_factor.max(1.0);

    let extra_space = (flow_parent_size - (sum_fit_content + min_sum_preferred + min_sum_grow)).max(0.0);

    // Calculate sizes for `FitContent`-sizing entities
    for (idx, child) in children_sizings.iter().enumerate() {
        if matches!(&child.sizing, Sizing::FitContent | Sizing::ParentBased(_)) {
            result[idx].1 = child.min_flow_size;
        }
    }

    let mut sum_preferred = 0.0;
    let mut preferred_space_left = min_sum_preferred + extra_space;

    // Calculate sizes for `Preferred`-sizing entities
    for (idx, child) in children_sizings.iter().enumerate() {
        if child.relatively_positioned {
            continue;
        }
        let new_size = if let Sizing::Preferred(size) = child.sizing {
            size.min(preferred_space_left)
                .clamp(child.min_flow_size, child.max_flow_size)
        } else {
            continue;
        };
        sum_preferred += new_size;
        preferred_space_left -= new_size;
        result[idx].1 = new_size;
    }

    let space_for_grow = (flow_parent_size - (sum_fit_content + sum_preferred)).max(0.0);
    let mut grow_space_left = space_for_grow;

    // Calculate sizes for `Grow`-sizing entities
    for (idx, child) in children_sizings.iter().enumerate() {
        if child.relatively_positioned {
            if let Sizing::Grow(factor) = child.sizing {
                result[idx].1 =
                    (parent_size[flow_axis] * factor).clamp(child.min_flow_size, child.max_flow_size);
            }
            continue;
        }
        let size = if let Sizing::Grow(factor) = child.sizing {
            let proportion = factor / sum_grow_factor;
            let grow_size = proportion * space_for_grow;

            grow_size
                .min(grow_space_left)
                .clamp(child.min_flow_size, child.max_flow_size)
        } else {
            continue;
        };
        grow_space_left -= size;
        result[idx].1 = size;
    }

    result
}

struct ChildPositioningInfo {
    entity: EntityId,
    positioning: Position,
    cross_align: CrossAlign,
    size: Vec2,
}

/// Calculates relative position for each child in a flow.
fn flow_calculate_children_positions<F: FnMut(EntityId, Vec2)>(
    parent_size: Vec2,
    parent_padding: Padding,
    flow: ContentFlow,
    flow_align: FlowAlign,
    children_sizes: &[ChildPositioningInfo],
    mut output: F,
) {
    let flow_axis = flow.axis();
    let cross_flow_axis = flow.cross_axis();
    let flow_size_sum: f32 = children_sizes
        .iter()
        .filter(|child| child.positioning == Position::Auto)
        .map(|child| child.size[flow_axis])
        .sum();

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

        match child.positioning {
            Position::Auto => {
                pos[flow_axis] = curr_flow_pos;
                pos[cross_flow_axis] = cross_flow_pos;
            }
            Position::Relative(offset) => {
                pos = offset;
            }
        }

        pos.x += parent_padding.left;
        pos.y += parent_padding.top;

        output(child.entity, pos);

        if child.positioning == Position::Auto {
            curr_flow_pos += child.size[flow_axis];
        }
    }
}

impl UIRenderer {
    pub fn new(ctx: &EngineContext, root_entity: EntityId) -> Self {
        let mut scene = ctx.module_mut::<Scene>();
        let ui_layout_changes = scene.change_manager_mut().register_component_flow::<UILayoutC>();
        let relation_changes = scene.change_manager_mut().register_component_flow::<Relation>();

        let root_ui_entity = scene
            .add_object(Some(root_entity), UIObject::new_raw(UILayoutC::column(), ()))
            .unwrap();

        Self {
            root_ui_entity,
            root_element_size: Default::default(),
            root_element_size_dirty: false,
            scale_factor: 1.0,
            ui_layout_changes,
            relation_changes,
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
        scene: &mut Scene,
        dirty_elements: &mut HashSet<EntityId>,
    ) {
        const ROOT_IDX: usize = 0;

        // Start from children (bottom of the tree)
        for (i, node) in linear_tree.iter().enumerate().rev() {
            if !dirty_elements.contains(node) {
                continue;
            }

            let entry = scene.storage().entry(node).unwrap();
            let Some(layout) = entry.get::<UILayoutC>() else {
                continue;
            };
            let relation = entry.get::<Relation>().unwrap();
            let parent_entity = relation.parent;
            let flow_axis = layout.content_flow.axis();
            let cross_flow_axis = layout.content_flow.cross_axis();
            let mut new_min_size = Vec2::from_element(0.0);

            // Calculate self min size using children min sizes
            for child in relation.ordered_children() {
                let child_entry = scene.storage().entry(&child).unwrap();
                let Some(child_layout) = child_entry.get::<UILayoutC>() else {
                    continue;
                };
                if &child_layout.position != &Position::Auto {
                    continue;
                }
                let child_cache = child_entry.get::<UILayoutCacheC>().unwrap();

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

            let mut entry = scene.entry(node);
            let cache = entry.get_mut::<UILayoutCacheC>();

            if cache.final_min_size == new_min_size {
                continue;
            }
            cache.final_min_size = new_min_size;

            if i != ROOT_IDX {
                // Make parent dirty
                dirty_elements.insert(parent_entity);
            }
        }
    }

    /// For each element calculates expanded size and its position (starting from the top of the tree).
    /// Also hierarchically calculates final visibilities.
    fn perform_expansion(
        linear_tree: &[EntityId],
        scene: &mut Scene,
        dirty_elements: &mut HashSet<EntityId>,
        ctx: &EngineContext,
    ) {
        // Set final size for the root element
        {
            let mut root = scene.entry(&linear_tree[0]);
            let root_cache = root.get_mut::<UILayoutCacheC>();
            root_cache.final_size = root_cache.final_min_size;
            root_cache.clip_rect = ClipRect {
                min: Vec2::from_element(0.0),
                max: root_cache.final_size,
            };
        }

        // Start from top of the tree
        for node in linear_tree {
            let entry = scene.entry(node);

            let Some(relation) = entry.get_checked::<Relation>() else {
                continue;
            };

            let Some(&parent_layout) = entry.get_checked::<UILayoutC>() else {
                continue;
            };
            let children: SmallVec<[EntityId; 128]> = relation.ordered_children().collect();

            let contains_dirty_children = children.iter().any(|child| dirty_elements.contains(child));
            if !dirty_elements.contains(node) && !contains_dirty_children {
                continue;
            }

            let parent_cache = *entry.get::<UILayoutCacheC>();
            let flow_axis = parent_layout.content_flow.axis();
            let cross_flow_axis = parent_layout.content_flow.cross_axis();
            let parent_size = glm::max(&(parent_cache.final_size - parent_layout.padding.size()), 0.0);
            let parent_clip = &parent_cache.clip_rect;

            let mut children_flow_sizings: SmallVec<[ChildFlowSizingInfo; 128]> =
                SmallVec::with_capacity(relation.num_children());

            drop(entry);

            for child in &children {
                let mut child_entry = scene.entry(child);
                let Some(&child_layout) = child_entry.get_checked::<UILayoutC>() else {
                    continue;
                };
                let child_cache = *child_entry.get::<UILayoutCacheC>();

                let child_cross_min_size = child_layout.constraints[cross_flow_axis]
                    .min
                    .max(child_cache.final_min_size[cross_flow_axis]);
                let child_cross_max_size = child_layout.constraints[cross_flow_axis]
                    .max
                    .max(child_cross_min_size);

                // Calculate cross-flow-size for `child`
                let child_new_cross_size = match child_layout.sizing[cross_flow_axis] {
                    Sizing::Preferred(size) => size.clamp(
                        child_cache.final_min_size[cross_flow_axis],
                        parent_size[cross_flow_axis],
                    ),
                    Sizing::FitContent => child_cache.final_min_size[cross_flow_axis],
                    Sizing::Grow(factor) => parent_size[cross_flow_axis] * factor.min(1.0),
                    Sizing::ParentBased(calc_func) => calc_func(&child_entry, ctx, &parent_size),
                }
                .clamp(child_cross_min_size, child_cross_max_size);

                // Set new cross-flow size
                if child_new_cross_size != child_cache.final_size[cross_flow_axis] {
                    let child_cache = child_entry.get_mut::<UILayoutCacheC>();
                    child_cache.final_size[cross_flow_axis] = child_new_cross_size;
                    dirty_elements.insert(*child);
                }

                // if child_layout.content_transform.offset.x < 0.0 {
                //     dbg!(
                //         child_new_cross_size,
                //         child_layout.constraints[cross_flow_axis],
                //         child_cache.final_min_size
                //     );
                // }

                let max_flow_size = child_layout.constraints[flow_axis].max;
                let min_flow_size = if let Sizing::ParentBased(calc_func) = &child_layout.sizing[flow_axis] {
                    child_layout.constraints[flow_axis].clamp(calc_func(&child_entry, ctx, &parent_size))
                } else {
                    child_cache.final_min_size[flow_axis]
                };

                if child_layout.visibility == Visibility::Collapsed {
                    continue;
                }

                // Collect info for final size calculation
                children_flow_sizings.push(ChildFlowSizingInfo {
                    entity: *child,
                    min_flow_size,
                    max_flow_size,
                    relatively_positioned: child_layout.position != Position::Auto,
                    sizing: child_layout.sizing[flow_axis],
                    overflow: child_layout.overflow[flow_axis],
                });
            }

            // Calculate flow-size of each child
            let calculated_children_flow_sizes =
                flow_calculate_children_sizes(parent_size, flow_axis, &children_flow_sizings);

            let mut children_positioning_infos: SmallVec<[ChildPositioningInfo; 128]> =
                SmallVec::with_capacity(children_flow_sizings.len());

            // Set children flow sizes
            for (child, child_new_flow_size) in &calculated_children_flow_sizes {
                let mut child_entry = scene.entry(child);
                let child_layout_positioning = child_entry.get::<UILayoutC>().position;
                let child_layout_align = child_entry.get::<UILayoutC>().align;
                let child_cache = child_entry.get_mut::<UILayoutCacheC>();

                if *child_new_flow_size != child_cache.final_size[flow_axis] {
                    child_cache.final_size[flow_axis] = *child_new_flow_size;
                    dirty_elements.insert(*child);
                }

                // Collect info for position calculation
                children_positioning_infos.push(ChildPositioningInfo {
                    entity: *child,
                    positioning: child_layout_positioning,
                    cross_align: child_layout_align,
                    size: child_cache.final_size,
                });
            }

            // Calculate position of each child
            flow_calculate_children_positions(
                parent_size,
                parent_layout.padding,
                parent_layout.content_flow,
                parent_layout.flow_align,
                &children_positioning_infos,
                |child, position| {
                    let mut child_entry = scene.entry(&child);
                    let ui_transform = child_entry.get_mut::<UILayoutC>().content_transform;
                    let child_cache = child_entry.get_mut::<UILayoutCacheC>();

                    child_cache.relative_position = position + ui_transform.offset;
                    let new_global_pos = parent_cache.global_position + child_cache.relative_position;

                    if child_cache.global_position != new_global_pos {
                        child_cache.global_position = new_global_pos;
                        dirty_elements.insert(child);
                    }
                },
            );

            // Calculate clipping regions
            for (child, child_final_flow_size) in &calculated_children_flow_sizes {
                let mut child_entry = scene.entry(&child);
                let ui_transform = child_entry.get::<UILayoutC>().content_transform;
                let child_cache = child_entry.get_mut::<UILayoutCacheC>();

                let mut local_clip_rect = ClipRect::default();
                local_clip_rect.min = child_cache.relative_position;
                local_clip_rect.max = local_clip_rect.min;
                local_clip_rect.max[cross_flow_axis] += child_cache.final_size[cross_flow_axis];
                local_clip_rect.max[flow_axis] += child_final_flow_size;

                let global_clip_rect = ClipRect {
                    min: parent_cache.global_position + local_clip_rect.min - ui_transform.offset,
                    max: parent_cache.global_position + local_clip_rect.max - ui_transform.offset,
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
                for child in relation.unordered_children() {
                    dirty_elements.insert(child);
                }
            }
        }
    }

    fn handle_ui_layout_updates(scene: &mut Scene, dirty_elements: &HashSet<EntityId>) {
        // Calculate new final visibilities
        for entity_id in dirty_elements {
            let mut entry = scene.entry(entity_id);

            let layout = entry.get::<UILayoutC>();
            let visible = layout.visibility.is_visible();

            let relation = entry.get_mut::<Relation>();
            relation.active = visible;
        }
    }

    /// Outputs objects that contain the specified point to the specified closure.
    /// If the closure returns `true` the traversal is stopped. Iterates children first.
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

        let mut dirty_elements: HashSet<_> = scene
            .change_manager_mut()
            .take_new_iter(self.ui_layout_changes)
            .collect();

        // Relation changes are also important because children order can change
        let dirty_relations: Vec<_> = scene.change_manager_mut().take_new(self.relation_changes);
        dirty_elements.extend(
            dirty_relations
                .iter()
                .filter(|entity| scene.entry(entity).get_checked::<UILayoutC>().is_some()),
        );

        if !self.force_update_transforms && dirty_elements.is_empty() {
            return;
        }

        Self::handle_ui_layout_updates(&mut scene, &dirty_elements);

        let linear_tree =
            common::scene::collect_relation_tree(&scene.storage_mut().access(), &self.root_ui_entity);

        Self::calculate_final_minimum_sizes(&linear_tree, &mut scene, &mut dirty_elements);
        Self::perform_expansion(&linear_tree, &mut scene, &mut dirty_elements, ctx);

        Self::calculate_transforms(
            &linear_tree,
            &mut scene,
            &self.root_element_size,
            &mut dirty_elements,
            self.force_update_transforms,
        );

        // Dispatch on_size_update callbacks
        ctx.dispatch_callback(move |ctx, _| {
            let mut scene = ctx.module_mut::<Scene>();
            let callbacks: Vec<_> = dirty_elements
                .iter()
                .filter_map(|entity| {
                    let handler = scene
                        .entry(entity)
                        .get_checked::<UIEventHandlerC>()
                        .map(|v| v.on_size_update.as_ref().map(|v| Arc::clone(v)))
                        .flatten()?;
                    Some((*entity, handler))
                })
                .collect();
            drop(scene);

            for (entity, on_size_update) in callbacks {
                let new_size = {
                    let mut scene = ctx.module_mut::<Scene>();
                    let entry = scene.entry(&entity);
                    entry.get::<UILayoutCacheC>().final_size
                };
                on_size_update(&entity, ctx, new_size);
            }
        });

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
