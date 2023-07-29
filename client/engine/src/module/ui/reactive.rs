use crate::utils::transition::{AnimatedValue, Interpolatable};
use crate::EngineContext;
use common::parking_lot::Mutex;
use common::types::{HashMap, HashSet};
use std::any::Any;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync;
use std::sync::Arc;

pub type ScopeId = String;
pub type StateId = String;

pub struct UIScope {
    children_func: Rc<dyn Fn(&mut UIScopeContext)>,
    on_remove_func: Option<Box<dyn FnOnce(&EngineContext, &UIScope)>>,
    children: Vec<ScopeId>,
    states: HashMap<StateId, Arc<dyn Any + Send + Sync>>,
    local_vars: HashMap<String, Arc<dyn Any + Send + Sync>>,
    subscribed_to: HashSet<StateUID>,
}

impl UIScope {
    pub fn state<T: Send + Sync + 'static>(&self, name: &str) -> Option<Arc<T>> {
        self.states.get(name).map(|v| v.clone().downcast::<T>().unwrap())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct StateUID(ScopeId, StateId);

impl StateUID {
    pub fn scope_id(&self) -> &ScopeId {
        &self.0
    }

    pub fn state_id(&self) -> &StateId {
        &self.1
    }
}

type AnyState = Arc<dyn Any + Send + Sync>;
type ModifierFunctions = Vec<Box<dyn FnOnce(&AnyState) -> AnyState + Sync + Send>>;
type ModifiedStates = HashMap<StateUID, ModifierFunctions>;

/// Provides reactive ui management
pub struct UIReactor {
    scopes: HashMap<ScopeId, UIScope>,
    modified_states: Arc<Mutex<ModifiedStates>>,
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
                local_vars: Default::default(),
                subscribed_to: Default::default(),
            },
        );
        this.dirty_scopes.insert(root_id);
        this
    }

    pub fn on_update(&mut self, ctx: EngineContext, delta_time: f64) {
        let mut dirty_scopes: HashSet<_> = self.dirty_scopes.drain().collect();

        let modified_states: Vec<_> = self.modified_states.lock().drain().collect();

        // Update states and collect corresponding subscribers
        for (uid, modifiers) in modified_states {
            let Some(scope) = self
                .scopes
                .get_mut(uid.scope_id()) else {
                continue;
            };
            let state_value = scope.states.get_mut(uid.state_id()).unwrap();

            for modifier in modifiers {
                *state_value = modifier(state_value);
            }

            if let Some(subs) = self.state_subscribers.get(&uid) {
                dirty_scopes.extend(subs.iter().cloned());
            }
        }

        for scope_id in dirty_scopes {
            self.rebuild(scope_id, ctx, delta_time);
        }
    }

    pub fn local_var<T: Sync + Send + 'static>(
        &mut self,
        scope_id: &ScopeId,
        name: impl Into<String>,
        default_value: T,
    ) -> Arc<T> {
        let scope = self.scopes.get_mut(scope_id).unwrap();
        let val = scope
            .local_vars
            .entry(name.into())
            .or_insert_with(|| Arc::new(default_value));
        val.clone().downcast::<T>().unwrap()
    }

    pub fn set_local_var<T: Sync + Send + 'static>(
        &mut self,
        scope_id: &ScopeId,
        name: impl Into<String>,
        value: T,
    ) {
        let scope = self.scopes.get_mut(scope_id).unwrap();
        scope.local_vars.insert(name.into(), Arc::new(value));
    }

    pub fn contains_state<T: Send + Sync + 'static>(&self, state: &ReactiveState<T>) -> bool {
        self.scopes
            .get(&state.owner)
            .is_some_and(|scope| scope.states.contains_key(&state.name))
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
            modified_states: Arc::downgrade(&self.modified_states),
            value,
            _ty: Default::default(),
        })
    }

    /// Performs rebuilding of the specified scope of `parent`.
    fn rebuild(&mut self, scope_id: ScopeId, ctx: EngineContext, dt: f64) {
        let Some(scope) = self.scopes.get_mut(&scope_id) else {
            return;
        };
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

        for scope_id in queue.into_iter().rev() {
            let mut scope = self.scopes.remove(&scope_id).unwrap();
            let on_remove = scope.on_remove_func.take().unwrap();
            on_remove(ctx, &scope);

            self.dirty_scopes.remove(&scope_id);

            for state_id in scope.states.keys() {
                if let Some(subscribers) = self
                    .state_subscribers
                    .get_mut(&StateUID(scope_id.clone(), state_id.clone()))
                {
                    subscribers.remove(&scope_id);
                }
            }
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
            ctx.reactor.modified_states.lock().remove(&state_uid);

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
    modified_states: sync::Weak<Mutex<ModifiedStates>>,
    value: Arc<T>,
    _ty: PhantomData<T>,
}

impl<T: Send + Sync + 'static> ReactiveState<T> {
    fn uid(&self) -> StateUID {
        StateUID(self.owner.clone(), self.name.clone())
    }

    /// Enqueues a new state update.
    pub fn update_with<F: FnOnce(&T) -> T + Send + Sync + 'static>(&self, new_state_fn: F) {
        let Some(modified_states) = self.modified_states.upgrade() else {
            return;
        };
        let mut modified_states = modified_states.lock();

        let modifiers = modified_states.entry(self.uid()).or_default();
        modifiers.push(Box::new(|prev| {
            Arc::new(new_state_fn(prev.downcast_ref::<T>().unwrap()))
        }));
    }

    /// Enqueues a new state update.
    pub fn update(&self, new_value: T) {
        self.update_with(move |_| new_value);
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> Clone for ReactiveState<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            name: self.name.clone(),
            modified_states: self.modified_states.clone(),
            value: Arc::clone(&self.value),
            _ty: Default::default(),
        }
    }
}

impl<T: Default> Default for ReactiveState<T> {
    fn default() -> Self {
        Self {
            owner: "".to_string(),
            name: "".to_string(),
            modified_states: Default::default(),
            value: Default::default(),
            _ty: Default::default(),
        }
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

    pub fn num_children(&self) -> usize {
        self.used_children.len()
    }

    pub fn set_local_var<T: Sync + Send + 'static>(&mut self, name: impl Into<String>, value: T) {
        self.reactor.set_local_var(&self.scope_id, name, value);
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
            modified_states: Arc::downgrade(&self.reactor.modified_states),
            value: state_value.clone().downcast::<T>().unwrap(),
            _ty: Default::default(),
        }
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

        if *value.state().value() != new_value {
            value.state().update(new_value);
        }
    }

    pub fn descendant_scope_id(&self, scope_name: &str) -> ScopeId {
        format!("{}_{}", &self.scope_id, scope_name)
    }

    /// Performs descent of `scope` if the `parent` hasn't been descended earlier.
    pub fn descend<F, R>(&mut self, scope_name: &str, children_fn: F, on_remove_fn: R, force_rebuild: bool)
    where
        F: Fn(&mut UIScopeContext) + 'static,
        R: FnOnce(&EngineContext, &UIScope) + 'static,
    {
        let scope_id = self.descendant_scope_id(scope_name);
        self.used_children.push(scope_id.clone());

        if !self.reactor.scopes.contains_key(&scope_id) {
            let parent_scope = self.reactor.scopes.get(&self.scope_id).unwrap();
            let parent_local_vars = parent_scope.local_vars.clone();

            self.reactor.scopes.insert(
                scope_id.clone(),
                UIScope {
                    children_func: Rc::new(children_fn),
                    on_remove_func: Some(Box::new(on_remove_fn)),
                    children: vec![],
                    states: Default::default(),
                    local_vars: parent_local_vars,
                    subscribed_to: Default::default(),
                },
            );
        } else {
            let e = self.reactor.scopes.get_mut(&scope_id).unwrap();
            e.children_func = Rc::new(children_fn);

            if !force_rebuild {
                return;
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