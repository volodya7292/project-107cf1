use crate::EngineContext;
use common::types::{HashMap, HashSet};
use entity_data::EntityId;
use std::any::Any;
use std::cell::RefCell;
use std::collections::hash_map;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

struct UIScope {
    children_func: Rc<dyn Fn(&mut UIScopeContext)>,
    on_remove_func: Box<dyn FnOnce(&EngineContext)>,
    children: Vec<EntityId>,
    states: HashMap<&'static str, RefCell<Rc<dyn Any>>>,
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

    /// Performs rebuilding of the specified scope of `parent`.
    fn rebuild(&mut self, parent: EntityId, ctx: EngineContext) {
        let scope = self.scopes.get(&parent).unwrap();
        let func = Rc::clone(&scope.children_func);

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

pub struct UIScopeContext<'a, 'b> {
    ctx: EngineContext<'a>,
    reactor: &'b mut UIReactor,
    parent: EntityId,
    used_children: Vec<EntityId>,
    used_states: HashSet<&'static str>,
}

#[derive(Clone)]
pub struct State<T> {
    owner: EntityId,
    name: &'static str,
    value: Rc<T>,
    _ty: PhantomData<T>,
}

impl<T> State<T> {
    fn uid(&self) -> StateUID {
        StateUID(self.owner, self.name)
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

pub struct StateReceiver<T> {
    value: Rc<T>,
    _ty: PhantomData<T>,
}

impl<T: 'static> Deref for StateReceiver<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<'a, 'b> UIScopeContext<'a, 'b> {
    pub fn ctx(&self) -> &EngineContext<'a> {
        &self.ctx
    }

    pub fn parent(&self) -> EntityId {
        self.parent
    }

    pub fn request_state<T: 'static, F: FnOnce() -> T>(&mut self, name: &'static str, init: F) -> State<T> {
        let scope = self.reactor.scopes.get_mut(&self.parent).unwrap();

        let state_value = scope
            .states
            .entry(name)
            .or_insert_with(|| RefCell::new(Rc::new(init())));

        State {
            owner: self.parent,
            name,
            value: state_value.borrow().clone().downcast::<T>().unwrap(),
            _ty: Default::default(),
        }
    }

    pub fn set_state<T: 'static>(&mut self, state: &State<T>, new_value: T) {
        let state_scope = self.reactor.scopes.get_mut(&state.owner).unwrap();
        let state_value = state_scope.states.get(state.name).unwrap();
        *state_value.borrow_mut() = Rc::new(new_value);
        self.reactor.modified_states.insert(state.uid());
    }

    pub fn subscribe<T: 'static>(&mut self, state: &State<T>) -> StateReceiver<T> {
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
    pub fn descend<F, R>(&mut self, parent: EntityId, scope_fn: F, on_remove_fn: R)
    where
        F: Fn(&mut UIScopeContext) + 'static,
        R: FnOnce(&EngineContext) + 'static,
    {
        self.used_children.push(parent);

        if let hash_map::Entry::Vacant(e) = self.reactor.scopes.entry(parent) {
            e.insert(UIScope {
                children_func: Rc::new(scope_fn),
                on_remove_func: Box::new(on_remove_fn),
                children: vec![],
                states: Default::default(),
                subscribed_to: Default::default(),
            });
        } else {
            return;
        };

        self.reactor.rebuild(parent, self.ctx);
    }
}

/// Requests a state and subscribes to it creating a variable with respective name.
macro_rules! remember_state {
    ($ctx: expr, $name: ident, $init: expr) => {
        let $name = $ctx.request_state(stringify!($name), || $init);
        let $name = $ctx.subscribe(&$name);
    };
}
