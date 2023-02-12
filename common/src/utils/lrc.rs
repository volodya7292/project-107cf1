use std::cell::{Ref, RefCell, RefMut};
use std::mem;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

/// Single-threaded locked reference-counted pointer
pub type Lrc<T> = Rc<RefCell<T>>;

pub trait LrcExtSized<T> {
    fn wrap(v: T) -> Self;
}

pub trait LrcExt<T: ?Sized> {
    fn borrow_owned(&self) -> OwnedRef<T, T>;
    fn borrow_mut_owned(&self) -> OwnedRefMut<T, T>;
}

impl<T> LrcExtSized<T> for Lrc<T> {
    fn wrap(v: T) -> Self {
        Rc::new(RefCell::new(v))
    }
}

impl<T: ?Sized> LrcExt<T> for Lrc<T> {
    fn borrow_owned(&self) -> OwnedRef<T, T> {
        let borrowed = self.borrow();
        let ref_unbounded = unsafe { mem::transmute::<_, Ref<'static, T>>(borrowed) };

        OwnedRef {
            owner: ManuallyDrop::new(Rc::clone(self)),
            reference: ManuallyDrop::new(ref_unbounded),
        }
    }

    fn borrow_mut_owned(&self) -> OwnedRefMut<T, T> {
        let borrowed = self.borrow_mut();
        let ref_unbounded = unsafe { mem::transmute::<_, RefMut<'static, T>>(borrowed) };

        OwnedRefMut {
            owner: ManuallyDrop::new(Rc::clone(self)),
            reference: ManuallyDrop::new(ref_unbounded),
        }
    }
}

pub struct OwnedRef<T: ?Sized + 'static, R: ?Sized + 'static> {
    owner: ManuallyDrop<Lrc<T>>,
    reference: ManuallyDrop<Ref<'static, R>>,
}

impl<T: ?Sized, R: ?Sized> OwnedRef<T, R> {
    pub fn map<U, F: FnOnce(&R) -> &U>(mut orig: OwnedRef<T, R>, f: F) -> OwnedRef<T, U> {
        let owner = unsafe { ManuallyDrop::take(&mut orig.owner) };
        let reference = unsafe { ManuallyDrop::take(&mut orig.reference) };
        let new_reference = Ref::map(reference, f);

        mem::forget(orig);

        OwnedRef {
            owner: ManuallyDrop::new(owner),
            reference: ManuallyDrop::new(new_reference),
        }
    }
}

impl<T: ?Sized, R: ?Sized> Deref for OwnedRef<T, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.reference.deref()
    }
}

impl<T: ?Sized, R: ?Sized> Drop for OwnedRef<T, R> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.reference);
            ManuallyDrop::drop(&mut self.owner);
        }
    }
}

pub struct OwnedRefMut<T: ?Sized + 'static, R: ?Sized + 'static> {
    owner: ManuallyDrop<Lrc<T>>,
    reference: ManuallyDrop<RefMut<'static, R>>,
}

impl<T: ?Sized, R: ?Sized> OwnedRefMut<T, R> {
    pub fn map<U, F: FnOnce(&mut R) -> &mut U>(mut orig: OwnedRefMut<T, R>, f: F) -> OwnedRefMut<T, U> {
        let owner = unsafe { ManuallyDrop::take(&mut orig.owner) };
        let reference = unsafe { ManuallyDrop::take(&mut orig.reference) };
        let new_reference = RefMut::map(reference, f);

        mem::forget(orig);

        OwnedRefMut {
            owner: ManuallyDrop::new(owner),
            reference: ManuallyDrop::new(new_reference),
        }
    }
}

impl<T: ?Sized, R: ?Sized> Deref for OwnedRefMut<T, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.reference.deref()
    }
}

impl<T: ?Sized, R: ?Sized> DerefMut for OwnedRefMut<T, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.reference.deref_mut()
    }
}

impl<T: ?Sized, R: ?Sized> Drop for OwnedRefMut<T, R> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.reference);
            ManuallyDrop::drop(&mut self.owner);
        }
    }
}
