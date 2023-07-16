use crate::EngineContext;
use common::types::{HashMap, HashSet};
use entity_data::EntityId;
use std::any::Any;
use std::cell::RefCell;
use std::collections::hash_map;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

struct UIScope {
    children_func: Rc<dyn Fn(&mut UIScopeContext)>,
    on_remove_func: Box<dyn FnOnce(&EngineContext)>,
    children: Vec<EntityId>,
    states: HashMap<&'static str, RefCell<Arc<dyn Any + Send + Sync>>>,
    subscribed_to: HashSet<StateUID>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct StateUID(EntityId, &'static str);

/// Provides reactive ui management
pub struct UIReactor {
    scopes: HashMap<EntityId, UIScope>,
    modified_states: HashSet<StateUID>,
    dirty_scopes: HashSet<EntityId>,
    state_subscribers: HashMap<StateUID, HashSet<EntityId>>,
}

impl UIReactor {
    pub fn new<F>(root: EntityId, root_scope_fn: F) -> Self
    where
        F: Fn(&mut UIScopeContext) + 'static,
    {
        let mut this = Self {
            scopes: Default::default(),
            modified_states: Default::default(),
            dirty_scopes: Default::default(),
            state_subscribers: Default::default(),
        };
        this.scopes.insert(
            root,
            UIScope {
                children_func: Rc::new(root_scope_fn),
                on_remove_func: Box::new(|_| {}),
                children: vec![],
                states: Default::default(),
                subscribed_to: Default::default(),
            },
        );
        this.dirty_scopes.insert(root);
        this
    }

    pub fn on_update(&mut self, ctx: EngineContext) {
        let dirty_scopes: HashSet<_> = self
            .dirty_scopes
            .drain()
            .chain(
                self.modified_states
                    .drain()
                    .filter_map(|uid| self.state_subscribers.get(&uid))
                    .map(|uid| uid.iter().cloned())
                    .flatten(),
            )
            .collect();

        for entity in dirty_scopes {
            self.rebuild(entity, ctx);
        }
    }

    pub fn set_state<T: Send + Sync + 'static, F: FnOnce(&T) -> T>(
        &mut self,
        state: &ReactiveState<T>,
        new: F,
    ) {
        let state_scope = self.scopes.get_mut(&state.owner).unwrap();
        let mut state_value = state_scope.states.get(state.name).unwrap();

        let mut curr_value = state_value.borrow_mut();
        let new_value = new(curr_value.downcast_ref::<T>().unwrap());

        *curr_value = Arc::new(new_value);
        self.modified_states.insert(state.uid());
    }

    /// Performs rebuilding of the specified scope of `parent`.
    fn rebuild(&mut self, parent: EntityId, ctx: EngineContext) {
        let mut scope = self.scopes.get_mut(&parent).unwrap();
        let func = Rc::clone(&scope.children_func);

        scope.subscribed_to.clear();

        let mut child_ctx = UIScopeContext {
            ctx,
            reactor: self,
            parent,
            used_children: vec![],
            used_states: Default::default(),
        };
        func(&mut child_ctx);

        Self::remove_unused_children(&mut child_ctx);
        Self::remove_unused_states(&mut child_ctx);
    }

    fn remove_unused_children(ctx: &mut UIScopeContext) {
        let scope = ctx.reactor.scopes.get_mut(&ctx.parent).unwrap();

        let unused_children: Vec<_> = scope
            .children
            .iter()
            .filter(|child| !ctx.used_children.contains(child))
            .cloned()
            .collect();

        scope.children.clear();
        scope.children.extend(&ctx.used_children);

        for child in unused_children {
            ctx.reactor.remove_element(child, &ctx.ctx);
        }
    }

    fn remove_unused_states(ctx: &mut UIScopeContext) {
        let scope = ctx.reactor.scopes.get_mut(&ctx.parent).unwrap();

        let states_to_remove: Vec<_> = scope
            .states
            .keys()
            .filter(|key| !ctx.used_states.contains(**key))
            .cloned()
            .collect();

        for name in &states_to_remove {
            scope.states.remove(name);
        }

        for name in states_to_remove {
            let state_uid = StateUID(ctx.parent, name);
            ctx.reactor.modified_states.remove(&state_uid);

            if let Some(subscribers) = ctx.reactor.state_subscribers.remove(&state_uid) {
                for subscriber in subscribers {
                    let subscriber_scope = ctx.reactor.scopes.get_mut(&subscriber).unwrap();
                    subscriber_scope.subscribed_to.remove(&state_uid);
                }
            }
        }
    }

    fn remove_element(&mut self, element: EntityId, ctx: &EngineContext) {
        let mut to_visit = vec![element];
        let mut queue = vec![];

        while let Some(next) = to_visit.pop() {
            queue.push(next);

            let scope = self.scopes.get(&next).unwrap();
            to_visit.extend(scope.children.iter().rev());
        }

        for elem in queue.into_iter().rev() {
            let scope = self.scopes.remove(&elem).unwrap();
            (scope.on_remove_func)(ctx);
            self.dirty_scopes.remove(&elem);
        }
    }
}

#[derive(Clone)]
pub struct ReactiveState<T> {
    owner: EntityId,
    name: &'static str,
    value: Arc<T>,
    _ty: PhantomData<T>,
}

impl<T> ReactiveState<T> {
    fn uid(&self) -> StateUID {
        StateUID(self.owner, self.name)
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

pub struct StateReceiver<T> {
    value: Arc<T>,
    _ty: PhantomData<T>,
}

impl<T: 'static> Deref for StateReceiver<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

pub struct UIScopeContext<'a, 'b> {
    ctx: EngineContext<'a>,
    reactor: &'b mut UIReactor,
    parent: EntityId,
    used_children: Vec<EntityId>,
    used_states: HashSet<&'static str>,
}

impl<'a, 'b> UIScopeContext<'a, 'b> {
    pub fn ctx(&self) -> &EngineContext<'a> {
        &self.ctx
    }

    pub fn reactor(&mut self) -> &mut UIReactor {
        self.reactor
    }

    pub fn parent(&self) -> EntityId {
        self.parent
    }

    /// Returns the state identified by `name`. If the state doesn't exist,
    /// creates a new one with the return value from `init` closure.
    pub fn request_state<T: Send + Sync + 'static, F: FnOnce() -> T>(
        &mut self,
        name: &'static str,
        init: F,
    ) -> ReactiveState<T> {
        let scope = self.reactor.scopes.get_mut(&self.parent).unwrap();
        self.used_states.insert(name);

        let state_value = scope
            .states
            .entry(name)
            .or_insert_with(|| RefCell::new(Arc::new(init())));

        ReactiveState {
            owner: self.parent,
            name,
            value: state_value.borrow().clone().downcast::<T>().unwrap(),
            _ty: Default::default(),
        }
    }

    /// Subscribes this scope to updates of the state. This scope will be rebuild it `state` changes.
    pub fn subscribe<T: 'static>(&mut self, state: &ReactiveState<T>) -> StateReceiver<T> {
        let scope = self.reactor.scopes.get_mut(&self.parent).unwrap();
        scope.subscribed_to.insert(state.uid());

        self.reactor
            .state_subscribers
            .entry(state.uid())
            .or_default()
            .insert(self.parent);

        StateReceiver {
            value: state.value.clone(),
            _ty: Default::default(),
        }
    }

    /// Performs descent of `scope` if the `parent` hasn't been descended earlier.
    pub fn descend<F, R>(&mut self, parent: EntityId, children_fn: F, on_remove_fn: R, force_rebuild: bool)
    where
        F: Fn(&mut UIScopeContext) + 'static,
        R: FnOnce(&EngineContext) + 'static,
    {
        self.used_children.push(parent);

        match self.reactor.scopes.entry(parent) {
            hash_map::Entry::Vacant(e) => {
                e.insert(UIScope {
                    children_func: Rc::new(children_fn),
                    on_remove_func: Box::new(on_remove_fn),
                    children: vec![],
                    states: Default::default(),
                    subscribed_to: Default::default(),
                });
            }
            hash_map::Entry::Occupied(mut e) => {
                e.get_mut().children_func = Rc::new(children_fn);

                if !force_rebuild {
                    return;
                }
            }
        }

        self.reactor.rebuild(parent, self.ctx);
    }
}

/// Requests a state and subscribes to it creating a variable with respective name.
#[macro_export]
macro_rules! remember_state {
    ($ctx: expr, $name: ident, $init: expr) => {
        let $name = $ctx.request_state(stringify!($name), || $init);
        let $name = $ctx.subscribe(&$name);
    };
}
