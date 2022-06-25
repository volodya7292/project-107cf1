use std::pin::Pin;
use std::ptr;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[cxx::bridge]
mod sys {
    #[derive(Default, Copy, Clone, PartialEq)]
    pub struct Vector2 {
        x: f64,
        y: f64,
    }

    #[namespace = "msdfgen"]
    unsafe extern "C++" {
        include!("lib.h");

        pub type Shape;
        pub type Contour;

        pub fn addContour(self: Pin<&mut Shape>) -> Pin<&mut Contour>;
        pub fn normalize(self: Pin<&mut Shape>);

        pub fn edgeColoringSimple(shape: Pin<&mut Shape>, angle_threshold: f64, seed: u64);
    }

    unsafe extern "C++" {
        pub type Vector2;

        pub fn create_shape() -> UniquePtr<Shape>;
        pub fn shape_check_last_contour(shape: Pin<&mut Shape>);

        pub unsafe fn contour_is_edges_empty(contour: *const Contour) -> bool;
        pub unsafe fn contour_add_edge2(contour: *mut Contour, p0: Vector2, p1: Vector2);
        pub unsafe fn contour_add_edge3(contour: *mut Contour, p0: Vector2, p1: Vector2, p2: Vector2);
        pub unsafe fn contour_add_edge4(
            contour: *mut Contour,
            p0: Vector2,
            p1: Vector2,
            p2: Vector2,
            p3: Vector2,
        );
        pub unsafe fn generateMSDF(output: *mut u8, width: u32, height: u32, shape: &Shape, range: f64);
    }
}

struct SysOutlineBuilder<'a> {
    position: sys::Vector2,
    shape: Pin<&'a mut sys::Shape>,
    contour: *mut sys::Contour,
}

fn point_from_f26dot6(x: f32, y: f32) -> sys::Vector2 {
    sys::Vector2 {
        x: x as f64 / 64.0,
        y: y as f64 / 64.0,
    }
}

impl rusttype::OutlineBuilder for SysOutlineBuilder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.contour.is_null() || unsafe { !sys::contour_is_edges_empty(self.contour) } {
            let contour = self.shape.as_mut().addContour();
            self.contour = unsafe { Pin::get_unchecked_mut(contour) };
        }
        self.position = point_from_f26dot6(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let endpoint = point_from_f26dot6(x, y);
        if endpoint != self.position {
            unsafe { sys::contour_add_edge2(self.contour, self.position, endpoint) };
            self.position = endpoint;
        }
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let control = point_from_f26dot6(x1, y1);
        let to = point_from_f26dot6(x, y);
        unsafe { sys::contour_add_edge3(self.contour, self.position, control, to) };
        self.position = to;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let control1 = point_from_f26dot6(x1, y1);
        let control2 = point_from_f26dot6(x2, y2);
        let to = point_from_f26dot6(x, y);
        unsafe { sys::contour_add_edge4(self.contour, self.position, control1, control2, to) };
        self.position = to;
    }

    fn close(&mut self) {}
}

pub fn generate_msdf(glyph: &rusttype::ScaledGlyph, width: u32, height: u32, range: f32) -> Option<Vec<u8>> {
    let mut shape = sys::create_shape();
    let mut builder = SysOutlineBuilder {
        position: Default::default(),
        shape: shape.pin_mut(),
        contour: ptr::null_mut(),
    };

    if !glyph.build_outline(&mut builder) {
        return None;
    }

    sys::shape_check_last_contour(shape.pin_mut());
    shape.pin_mut().normalize();
    sys::edgeColoringSimple(shape.pin_mut(), 3.0, 0);

    let data_len = (width * height * 3 * 4) as usize;
    let mut data = Vec::with_capacity(data_len);
    unsafe {
        sys::generateMSDF(data.as_mut_ptr(), width, height, &shape, range as f64);
        data.set_len(data_len);
    };

    Some(data)
}
