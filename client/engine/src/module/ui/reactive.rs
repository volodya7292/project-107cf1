use crate::utils::transition::{AnimatedValue, Interpolatable};
use crate::EngineContext;
use common::types::{HashMap, HashSet};
use std::any::Any;
use std::collections::hash_map;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

pub type ScopeId = String;
pub type StateId = String;

pub struct UIScope {
    children_func: Rc<dyn Fn(&mut UIScopeContext)>,
    on_remove_func: Option<Box<dyn FnOnce(&EngineContext, &UIScope)>>,
    children: Vec<ScopeId>,
    states: HashMap<StateId, Arc<dyn Any + Send + Sync>>,
    subscribed_to: HashSet<StateUID>,
}

impl UIScope {
    pub fn state<T: Send + Sync + 'static>(&self, name: &str) -> Option<Arc<T>> {
        self.states.get(name).map(|v| v.clone().downcast::<T>().unwrap())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct StateUID(ScopeId, StateId);

/// Provides reactive ui management
pub struct UIReactor {
    scopes: HashMap<ScopeId, UIScope>,
    modified_states: HashSet<StateUID>,
    dirty_scopes: HashSet<ScopeId>,
    state_subscribers: HashMap<StateUID, HashSet<ScopeId>>,
}

impl UIReactor {
    pub const ROOT_SCOPE_ID: &'static str = "root";

    pub fn new<F>(root_scope_fn: F) -> Self
    where
        F: Fn(&mut UIScopeContext) + 'static,
    {
        let root_id = Self::ROOT_SCOPE_ID.to_string();
        let mut this = Self {
            scopes: Default::default(),
            modified_states: Default::default(),
            dirty_scopes: Default::default(),
            state_subscribers: Default::default(),
        };
        this.scopes.insert(
            root_id.clone(),
            UIScope {
                children_func: Rc::new(root_scope_fn),
                on_remove_func: Some(Box::new(|_, _| {})),
                children: vec![],
                states: Default::default(),
                subscribed_to: Default::default(),
            },
        );
        this.dirty_scopes.insert(root_id);
        this
    }

    pub fn on_update(&mut self, ctx: EngineContext, delta_time: f64) {
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
            self.rebuild(entity, ctx, delta_time);
        }
    }

    pub fn get_state<T: Send + Sync + 'static>(
        &self,
        scope_id: ScopeId,
        state_id: StateId,
    ) -> Option<ReactiveState<T>> {
        let scope = self.scopes.get(&scope_id)?;
        let state = scope.states.get(&state_id)?;
        let value = state.clone().downcast::<T>().unwrap();

        Some(ReactiveState {
            owner: scope_id,
            name: state_id,
            value,
            _ty: Default::default(),
        })
    }

    pub fn set_state<T: Send + Sync + 'static, F: FnOnce(&T) -> T>(
        &mut self,
        state: &ReactiveState<T>,
        new: F,
    ) {
        let state_scope = self.scopes.get_mut(&state.owner).unwrap();
        let state_value = state_scope.states.get_mut(&state.name).unwrap();

        let new_value = new(state_value.downcast_ref::<T>().unwrap());

        *state_value = Arc::new(new_value);
        self.modified_states.insert(state.uid());
    }

    /// Performs rebuilding of the specified scope of `parent`.
    fn rebuild(&mut self, scope_id: ScopeId, ctx: EngineContext, dt: f64) {
        let scope = self.scopes.get_mut(&scope_id).unwrap();
        let func = Rc::clone(&scope.children_func);

        scope.subscribed_to.clear();

        let mut child_ctx = UIScopeContext {
            ctx,
            delta_time: dt,
            reactor: self,
            scope_id,
            used_children: vec![],
            used_states: Default::default(),
        };
        func(&mut child_ctx);

        Self::remove_unused_children(&mut child_ctx);
        Self::remove_unused_states(&mut child_ctx);
    }

    fn remove_element(&mut self, element: ScopeId, ctx: &EngineContext) {
        let mut to_visit = vec![element];
        let mut queue = vec![];

        while let Some(next) = to_visit.pop() {
            let scope = self.scopes.get(&next).unwrap();
            to_visit.extend(scope.children.iter().cloned().rev());

            queue.push(next);
        }

        for elem in queue.into_iter().rev() {
            let mut scope = self.scopes.remove(&elem).unwrap();
            let on_remove = scope.on_remove_func.take().unwrap();
            on_remove(ctx, &scope);

            self.dirty_scopes.remove(&elem);
        }
    }

    fn remove_unused_children(ctx: &mut UIScopeContext) {
        let scope = ctx.reactor.scopes.get_mut(&ctx.scope_id).unwrap();

        let unused_children: Vec<_> = scope
            .children
            .iter()
            .filter(|child| !ctx.used_children.contains(child))
            .cloned()
            .collect();

        scope.children.clear();
        scope.children.extend(ctx.used_children.iter().cloned());

        for child in unused_children {
            ctx.reactor.remove_element(child, &ctx.ctx);
        }
    }

    fn remove_unused_states(ctx: &mut UIScopeContext) {
        let scope = ctx.reactor.scopes.get_mut(&ctx.scope_id).unwrap();

        let states_to_remove: Vec<_> = scope
            .states
            .keys()
            .filter(|key| !ctx.used_states.contains(*key))
            .cloned()
            .collect();

        for name in &states_to_remove {
            scope.states.remove(name);
        }

        for name in states_to_remove {
            let state_uid = StateUID(ctx.scope_id.clone(), name);
            ctx.reactor.modified_states.remove(&state_uid);

            if let Some(subscribers) = ctx.reactor.state_subscribers.remove(&state_uid) {
                for subscriber in subscribers {
                    let subscriber_scope = ctx.reactor.scopes.get_mut(&subscriber).unwrap();
                    subscriber_scope.subscribed_to.remove(&state_uid);
                }
            }
        }
    }
}

pub struct ReactiveState<T> {
    owner: ScopeId,
    name: String,
    value: Arc<T>,
    _ty: PhantomData<T>,
}

impl<T> Clone for ReactiveState<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            name: self.name.clone(),
            value: Arc::clone(&self.value),
            _ty: Default::default(),
        }
    }
}

impl<T> ReactiveState<T> {
    fn uid(&self) -> StateUID {
        StateUID(self.owner.clone(), self.name.clone())
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

pub struct StateReceiver<T> {
    state: ReactiveState<T>,
    _ty: PhantomData<T>,
    _not_sync_not_send: PhantomData<*const u8>,
}

impl<T> StateReceiver<T> {
    pub fn state(&self) -> &ReactiveState<T> {
        &self.state
    }
}

impl<T: 'static> Deref for StateReceiver<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.state.value
    }
}

pub struct UIScopeContext<'a, 'b> {
    ctx: EngineContext<'a>,
    delta_time: f64,
    reactor: &'b mut UIReactor,
    scope_id: ScopeId,
    used_children: Vec<ScopeId>,
    used_states: HashSet<String>,
}

impl<'a, 'b> UIScopeContext<'a, 'b> {
    pub fn ctx(&self) -> &EngineContext<'a> {
        &self.ctx
    }

    pub fn reactor(&mut self) -> &mut UIReactor {
        self.reactor
    }

    pub fn scope_id(&self) -> &ScopeId {
        &self.scope_id
    }

    /// Returns the state identified by `name`. If the state doesn't exist,
    /// creates a new one with the return value from `init` closure.
    pub fn request_state<T: Send + Sync + 'static, F: FnOnce() -> T>(
        &mut self,
        name: impl Into<String>,
        init: F,
    ) -> ReactiveState<T> {
        let name = name.into();

        if self.used_states.contains(&name) {
            panic!("State '{name}' has already been requested!");
        } else {
            self.used_states.insert(name.clone());
        }

        let scope = self.reactor.scopes.get_mut(&self.scope_id).unwrap();

        let state_value = scope
            .states
            .entry(name.clone())
            .or_insert_with(|| Arc::new(init()));

        ReactiveState {
            owner: self.scope_id.clone(),
            name,
            value: state_value.clone().downcast::<T>().unwrap(),
            _ty: Default::default(),
        }
    }

    pub fn set_state<T: Send + Sync + 'static, F: FnOnce(&T) -> T>(
        &mut self,
        state: &ReactiveState<T>,
        new: F,
    ) {
        self.reactor.set_state(state, new);
    }

    /// Subscribes this scope to updates of the state. This scope will be rebuild it `state` changes.
    pub fn subscribe<T: Send + Sync + 'static>(&mut self, state: &ReactiveState<T>) -> StateReceiver<T> {
        let scope = self.reactor.scopes.get_mut(&self.scope_id).unwrap();
        scope.subscribed_to.insert(state.uid());

        self.reactor
            .state_subscribers
            .entry(state.uid())
            .or_default()
            .insert(self.scope_id.clone());

        let latest_value = &self.reactor.scopes[&state.owner].states[&state.name];

        let mut state = state.clone();
        state.value = latest_value.clone().downcast::<T>().unwrap();

        StateReceiver {
            state,
            _ty: Default::default(),
            _not_sync_not_send: Default::default(),
        }
    }

    pub fn drive_transition<T: Interpolatable + PartialEq + Send + Sync + 'static>(
        &mut self,
        value: &StateReceiver<AnimatedValue<T>>,
    ) {
        let mut new_value = **value;
        new_value.advance(self.delta_time);

        if value.state().value().current() != new_value.current() {
            self.set_state(value.state(), |_| new_value);
        }
    }

    /// Performs descent of `scope` if the `parent` hasn't been descended earlier.
    pub fn descend<F, R>(&mut self, scope_name: &str, children_fn: F, on_remove_fn: R, force_rebuild: bool)
    where
        F: Fn(&mut UIScopeContext) + 'static,
        R: FnOnce(&EngineContext, &UIScope) + 'static,
    {
        let scope_id = format!("{}_{}", &self.scope_id, scope_name);

        self.used_children.push(scope_id.clone());

        match self.reactor.scopes.entry(scope_id.clone()) {
            hash_map::Entry::Vacant(e) => {
                e.insert(UIScope {
                    children_func: Rc::new(children_fn),
                    on_remove_func: Some(Box::new(on_remove_fn)),
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

        self.reactor.rebuild(scope_id, self.ctx, self.delta_time);
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
