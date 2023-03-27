// use std::any::TypeId;
// use std::marker::PhantomData;
// use std::ptr;

// pub trait CtxFnOnce<Ctx>: FnOnce(&Ctx) + 'static {
//     /// Safety: `self` must not be used in any way after calling this function.
//     unsafe fn call(&mut self, ctx: &Ctx);
// }
//
// impl<Ctx, F: FnOnce(&Ctx) + 'static> CtxFnOnce<Ctx> for F {
//     unsafe fn call(&mut self, ctx: &Ctx) {
//         ptr::read(self)(ctx)
//     }
// }
//
// type DynCtxFnOnce<Ctx> = dyn CtxFnOnce<Ctx, Output = ()>;
//
// pub struct OnceFnStorage {
//     data: bumpalo::Bump,
//     TODO: add drop func pointer
//     ptrs: Vec<Option<*mut u8>>,
//     ctx_ty: TypeId,
// }
//
// impl OnceFnStorage {
//     pub fn new<Ctx>() -> Self {
//         Self {
//             data: bumpalo::Bump::with_capacity(1024),
//             ptrs: Vec::with_capacity(1024),
//             ctx_ty: TypeId::of::<Ctx>(),
//         }
//     }
//
//     /// Adds a function to the storage and returns its index.
//     pub fn add<Ctx>(&mut self, func: impl CtxFnOnce<Ctx, Output = ()>) -> usize {
//         assert_eq!(TypeId::of::<Ctx>(), self.ctx_ty);
//
//         let v = self.data.alloc(func);
//         self.ptrs.push(Some(v as *mut _));
//         self.ptrs.len() - 1
//     }
//
//     /// Calls the stored function at `index`.
//     pub fn call<Ctx>(&mut self, index: usize, ctx: &Ctx) {
//         assert_eq!(TypeId::of::<Ctx>(), self.ctx_ty);
//
//         let func_ptr = self
//             .ptrs
//             .get_mut(index)
//             .expect("function index must be valid")
//             .expect("function must be called only once");
//
//         let func = unsafe { &mut *func_ptr };
//         unsafe { func.call(ctx) };
//     }
//
//     pub fn clear(&mut self) {
//         self.ptrs.clear();
//         self.data.reset();
//     }
// }
//
// impl<Ctx> Drop for OnceFnStorage<Ctx> {
//     fn drop(&mut self) {
//         for &p in &self.ptrs {
//             if let Some(p) = p {
//                 unsafe {
//                     ptr::drop_in_place(p);
//                 }
//             }
//         }
//     }
// }
