use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::{
    ContainerProps, background, container, container_props, container_props_init,
};
use common::glm::Vec2;
use common::make_static_id;
use engine::ecs::component::ui::{ContentFlow, Padding, Position, Sizing, UILayoutC};
use engine::module::ui::reactive::UIScopeContext;
use engine::{EngineContext, remember_state};
use entity_data::EntityId;
use std::sync::Arc;

pub fn scrollable_container<P, F>(
    local_name: &str,
    ctx: &mut UIScopeContext,
    props: ContainerProps<P>,
    children_fn: F,
) where
    P: Clone + PartialEq + 'static,
    F: Fn(&mut UIScopeContext, P) + 'static,
{
    let children_fn = Arc::new(children_fn);

    container(
        local_name,
        ctx,
        container_props_init(props.clone()).layout(
            props
                .layout
                .with_content_flow(ContentFlow::Horizontal)
                .with_padding(Padding::ZERO),
        ),
        move |ctx, props| {
            let children_fn = Arc::clone(&children_fn);

            remember_state!(ctx, content_pos, 0.0_f32);
            remember_state!(ctx, wrapper_height, 0.0_f32);
            remember_state!(ctx, content_height, 0.0_f32);
            remember_state!(ctx, max_content_neg_offset, 0.0_f32);

            let new_max_content_neg_offset = (-*content_height + *wrapper_height).min(0.0);

            if *max_content_neg_offset != new_max_content_neg_offset {
                max_content_neg_offset.state().update(new_max_content_neg_offset);
            }

            if *content_pos < *max_content_neg_offset {
                content_pos.state().update(*max_content_neg_offset);
            }

            let on_wrapper_size_update = {
                let wrapper_height_state = wrapper_height.state();
                Arc::new(move |_: &EntityId, _: &EngineContext, new_size: Vec2| {
                    wrapper_height_state.update(new_size.y);
                })
            };
            let on_content_size_update = {
                let content_height_state = content_height.state();
                Arc::new(move |_: &EntityId, _: &EngineContext, new_size: Vec2| {
                    content_height_state.update(new_size.y);
                })
            };

            let on_scroll = {
                let vertical_pos_state = content_pos.state();
                let max_content_neg_offset_state = max_content_neg_offset.state();

                Arc::new(move |_: &EntityId, _: &EngineContext, delta: f64| {
                    let max_content_neg_offset_state = max_content_neg_offset_state.clone();
                    vertical_pos_state.update_with(move |prev| {
                        (*prev + delta as f32).clamp(*max_content_neg_offset_state.value(), 0.0)
                    });
                })
            };

            let bar_grow_ratio = *wrapper_height / *content_height;
            let bar_offset = -*content_pos / (*content_height - *wrapper_height).max(0.0001)
                * (*wrapper_height * (1.0 - bar_grow_ratio));

            let vertical_pos_state = content_pos.state();

            // content wrapper (window)
            container(
                make_static_id!(),
                ctx,
                container_props_init(props.clone())
                    .layout(UILayoutC::new().with_grow())
                    .callbacks(
                        props
                            .callbacks
                            .clone()
                            .on_scroll(on_scroll)
                            .on_size_update(on_wrapper_size_update),
                    ),
                move |ctx, props| {
                    let children_fn = Arc::clone(&children_fn);
                    let on_content_size_update = Arc::clone(&on_content_size_update);
                    let vertical_pos = ctx.subscribe(&vertical_pos_state);

                    // content
                    container(
                        make_static_id!(),
                        ctx,
                        props
                            .clone()
                            .layout(
                                props
                                    .layout
                                    .with_position(Position::Relative(Vec2::new(0.0, *vertical_pos)))
                                    .with_height(Sizing::FitContent),
                            )
                            .callbacks(props.callbacks.clone().on_size_update(on_content_size_update)),
                        move |ctx, props| {
                            children_fn(ctx, props);
                        },
                    );
                },
            );

            if bar_grow_ratio < 0.9999 {
                // bar
                container(
                    make_static_id!(),
                    ctx,
                    container_props()
                        .layout(UILayoutC::new().with_fixed_width(6.0).with_height_grow())
                        .children_props((bar_offset, bar_grow_ratio)),
                    move |ctx, (bar_offset, bar_grow_ratio)| {
                        container(
                            make_static_id!(),
                            ctx,
                            container_props()
                                .layout(
                                    UILayoutC::new()
                                        .with_position(Position::Relative(Vec2::new(0.0, bar_offset)))
                                        .with_width_grow()
                                        .with_height(Sizing::Grow(bar_grow_ratio))
                                        .with_padding(Padding::hv(2.0, 2.0)),
                                )
                                .background(Some(background::solid_color(DEFAULT_FANCY_COLOR)))
                                .corner_radius(4.0),
                            |_, ()| {},
                        );
                    },
                );
            }
        },
    );
}
