use crate::ecs::component;
use crate::ecs::component::internal::{GlobalTransform, Relation};
use crate::ecs::component::ui::{Position, Sizing, UILayout, UILayoutCache};
use crate::renderer::module::RendererModule;
use crate::renderer::{Internals, Renderer, SceneObject};
use entity_data::{Archetype, EntityId, SystemAccess};
use nalgebra_glm::DVec2;
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::any::Any;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use vk_wrapper::{CmdList, Device};

pub struct UIRenderer {
    device: Arc<Device>,
    root_ui_entity: EntityId,
    base_resolution: DVec2,
}

pub struct DefaultStyling {}

pub trait UIElement: Send + Sync + 'static {
    fn min_size() -> DVec2 {
        DVec2::default()
    }
}

impl UIElement for () {}

#[derive(Archetype)]
pub struct UIObject<E>
where
    E: UIElement,
{
    relation: Relation,
    global_transform: GlobalTransform,

    transform: component::Transform,
    renderer: component::MeshRenderConfig,
    mesh: component::VertexMesh,

    layout_cache: UILayoutCache,
    layout: UILayout,
    element: E,
}

impl<E: UIElement> UIObject<E> {
    pub fn new(layout: UILayout, element: E) -> Self {
        Self {
            relation: Default::default(),
            global_transform: Default::default(),
            transform: Default::default(),
            renderer: Default::default(),
            mesh: Default::default(),
            layout_cache: Default::default(),
            layout,
            element,
        }
    }

    pub(crate) fn with_layout_cache(mut self, layout_cache: UILayoutCache) -> Self {
        self.layout_cache = layout_cache;
        self
    }
}

impl<E: UIElement> SceneObject for UIObject<E> {}

struct ChildSize {
    id: EntityId,
    min: f64,
    sizing: Sizing,
}

/// Calculates final size for each child in a flow. Outputs pairs of [EntityId] and corresponding size.
fn calculate_children_sizes(
    parent_size: f64,
    children_sizings: &[ChildSize],
) -> impl Iterator<Item = (EntityId, f64)> + '_ {
    let min_sum: f64 = children_sizings.iter().map(|v| v.min).sum();
    let fit_parent_size = parent_size.max(min_sum);

    let mut sum_exact = 0.0;
    let mut sum_preferred = 0.0;
    let mut sum_grow_factor = 0.0;
    let mut max_space_for_preferred = fit_parent_size;

    for child in children_sizings {
        match child.sizing {
            Sizing::Exact(size) => sum_exact += size,
            Sizing::Grow(factor) => sum_grow_factor += factor,
            _ => {}
        }

        if let Sizing::Preferred(size) = child.sizing {
            sum_preferred += size;
        } else {
            max_space_for_preferred -= child.min;
        }
    }

    let max_space_for_grow = fit_parent_size - (sum_exact + max_space_for_preferred);
    let mut preferred_space_left = max_space_for_preferred;
    let mut grow_space_left = max_space_for_grow;

    children_sizings.iter().map(move |child| {
        let final_size = match child.sizing {
            Sizing::Exact(size) => size,
            Sizing::Preferred(size) => {
                let proportion = size / sum_preferred;
                let allocated = (max_space_for_preferred * proportion).clamp(child.min, preferred_space_left);
                preferred_space_left -= allocated;
                allocated
            }
            Sizing::Grow(factor) => {
                let proportion = factor / sum_grow_factor;
                let allocated = (max_space_for_grow * proportion).clamp(child.min, grow_space_left);
                grow_space_left -= allocated;
                allocated
            }
            Sizing::FitContent => child.min,
        };
        (child.id, final_size)
    })
}

impl UIRenderer {
    pub fn new(renderer: &mut Renderer) -> Self {
        let root_ui_entity = renderer.add_object(
            None,
            UIObject::new(component::ui::UILayout::new(), ()).with_layout_cache(UILayoutCache {
                intrinsic_min_size: Default::default(),
                // max_size_allowed_by_parent: DVec2::from_element(f64::INFINITY),
                final_size: Default::default(),
            }),
        );

        Self {
            device: Arc::clone(&renderer.device),
            root_ui_entity,
            base_resolution: Default::default(),
        }
    }

    pub fn set_base_resolution(&mut self, resolution: DVec2) {
        self.base_resolution = resolution;
    }

    fn update_hierarchy(&mut self, data: SystemAccess) {
        let relation_comps = data.component::<Relation>();
        let layout_comps = data.component::<UILayout>();
        let mut layout_cache_comps = data.component_mut::<UILayoutCache>();
        let transform_comps = data.component::<component::Transform>();

        let mut linear_tree = Vec::with_capacity(layout_comps.count_entities());
        linear_tree.push(self.root_ui_entity);
        core::scene::collect_children_recursively(&data, &self.root_ui_entity, &mut linear_tree);

        // Calculate minimum sizes starting from children (bottom of the tree)
        for node in linear_tree.iter().rev() {
            let layout = layout_comps.get(node).unwrap();
            let flow_axis = layout.content_flow().axis();
            let cross_flow_axis = layout.content_flow().cross_axis();
            let mut min_size = DVec2::from_element(0.0);

            // Calculate self min size
            for (axis, sizing) in layout.sizing().iter().enumerate() {
                if let Sizing::Exact(size) = *sizing {
                    min_size[axis] = size;
                    continue;
                }

                let Some(relation) = relation_comps.get(node) else {
                    continue;
                };

                // Calculate self min size using children min sizes
                for child in &relation.children {
                    let Some(child_layout) = layout_comps.get(child) else {
                        continue;
                    };
                    if child_layout.position() != &Position::Auto {
                        continue;
                    }
                    let Some(child_cache) = layout_cache_comps.get(child) else {
                        continue;
                    };

                    min_size[flow_axis] += child_cache.intrinsic_min_size[flow_axis];

                    let min_cross_size = &mut min_size[cross_flow_axis];
                    *min_cross_size = min_cross_size.max(child_cache.intrinsic_min_size[cross_flow_axis]);
                }
            }

            let cache = layout_cache_comps.get_mut(node).unwrap();
            cache.intrinsic_min_size = min_size;
        }

        // Expand sizes to maximum allowed sizes (starting from the top of the tree)
        for node in &linear_tree {
            let layout = layout_comps.get(node).unwrap();
            let cache = layout_cache_comps.get_mut(node).unwrap();
            let flow_axis = layout.content_flow().axis();
            let cross_flow_axis = layout.content_flow().cross_axis();

            let mut curr_size = cache.final_size;

            let Some(relation) = relation_comps.get(node) else {
                continue;
            };

            let mut children_flow_sizings: SmallVec<[ChildSize; 128]> =
                SmallVec::with_capacity(relation.children.len());

            for child in &relation.children {
                let Some(child_layout) = layout_comps.get(child) else {
                    continue;
                };
                let child_cache = layout_cache_comps.get_mut(child).unwrap();

                // Calculate cross-flow-size for `child`
                let child_final_cross_size = match child_layout.sizing()[cross_flow_axis] {
                    Sizing::Exact(size) => size,
                    Sizing::Preferred(size) => size.clamp(
                        child_cache.intrinsic_min_size[cross_flow_axis],
                        curr_size[cross_flow_axis],
                    ),
                    Sizing::FitContent => child_cache.intrinsic_min_size[cross_flow_axis],
                    Sizing::Grow(factor) => curr_size[cross_flow_axis] * factor.min(1.0),
                };
                child_cache.final_size[cross_flow_axis] = child_final_cross_size;

                // Collect flow sizing for further calculations
                children_flow_sizings.push(ChildSize {
                    id: *child,
                    min: child_cache.intrinsic_min_size[flow_axis],
                    sizing: child_layout.sizing()[flow_axis],
                });
            }

            // Calculate flow-size for `child`
            let calculated_children_sizes =
                calculate_children_sizes(curr_size[flow_axis], &children_flow_sizings);

            for (child, final_size) in calculated_children_sizes {
                let child_cache = layout_cache_comps.get_mut(&child).unwrap();
                child_cache.final_size[flow_axis] = final_size;
            }
        }
    }
}

impl RendererModule for UIRenderer {
    fn on_object_remove(&mut self, _id: &EntityId, _scene: Internals) {
        todo!()
    }

    fn on_update(&mut self, internals: Internals) -> Option<Arc<Mutex<CmdList>>> {
        let layout_changes = internals.dirty_comps.take_changes::<UILayout>();

        let access = internals.storage.access();

        self.update_hierarchy(access);

        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
