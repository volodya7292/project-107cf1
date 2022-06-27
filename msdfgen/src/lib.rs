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
        pub unsafe fn generateMSDF(output: *mut f32, width: u32, height: u32, shape: &Shape, range: f64);
    }
}

impl sys::Vector2 {
    pub fn from_f32(x: f32, y: f32) -> Self {
        Self {
            x: x as f64,
            y: y as f64,
        }
    }
}

struct SysOutlineBuilder<'a> {
    position: sys::Vector2,
    shape: Pin<&'a mut sys::Shape>,
    contour: *mut sys::Contour,
}

impl rusttype::OutlineBuilder for SysOutlineBuilder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.contour.is_null() || unsafe { !sys::contour_is_edges_empty(self.contour) } {
            let contour = self.shape.as_mut().addContour();
            self.contour = unsafe { Pin::get_unchecked_mut(contour) };
        }
        self.position = sys::Vector2::from_f32(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let endpoint = sys::Vector2::from_f32(x, y);
        if endpoint != self.position {
            unsafe { sys::contour_add_edge2(self.contour, self.position, endpoint) };
            self.position = endpoint;
        }
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let control = sys::Vector2::from_f32(x1, y1);
        let to = sys::Vector2::from_f32(x, y);
        unsafe { sys::contour_add_edge3(self.contour, self.position, control, to) };
        self.position = to;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let control1 = sys::Vector2::from_f32(x1, y1);
        let control2 = sys::Vector2::from_f32(x2, y2);
        let to = sys::Vector2::from_f32(x, y);
        unsafe { sys::contour_add_edge4(self.contour, self.position, control1, control2, to) };
        self.position = to;
    }

    fn close(&mut self) {}
}

pub fn generate_msdf(glyph: rusttype::Glyph, size: u32, range: f32) -> Option<Vec<f32>> {
    let scaled = glyph.scaled(rusttype::Scale::uniform(1.0));
    let bounds = scaled.exact_bounding_box()?;

    let render_scale = size as f32 / bounds.width().max(bounds.height());
    let scaled = scaled
        .into_unscaled()
        .scaled(rusttype::Scale::uniform(render_scale));

    let positioned = scaled.positioned(rusttype::Point { x: 0.0, y: 0.0 });

    dbg!(positioned.pixel_bounding_box());

    let mut shape = sys::create_shape();
    let mut builder = SysOutlineBuilder {
        position: Default::default(),
        shape: shape.pin_mut(),
        contour: ptr::null_mut(),
    };

    if !positioned.build_outline(&mut builder) {
        return None;
    }

    sys::shape_check_last_contour(shape.pin_mut());
    shape.pin_mut().normalize();
    sys::edgeColoringSimple(shape.pin_mut(), 3.0, 0);

    let data_len = (size * size * 3) as usize;
    let mut data = Vec::<f32>::with_capacity(data_len);
    unsafe {
        sys::generateMSDF(data.as_mut_ptr(), size, size, &shape, range as f64);
        data.set_len(data_len);
    };

    Some(data)
}

// #[test]
// fn gov() {
//     let f = std::fs::read("/Users/admin/Downloads/Romanesco-Regular.ttf").unwrap();
//     let font = rusttype::Font::try_from_bytes(&f).unwrap();
//     let g = font.glyph('A');
//     let msdf = generate_msdf(g, 64, 4.0).unwrap();
//
//     let mut msdf_b = vec![0_u8; 64 * 64 * 3];
//     let min = msdf.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
//     let max = msdf.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
//
//     for (f, u) in msdf.iter().zip(&mut msdf_b) {
//         *u = (*f * 255.0).clamp(0.0, 255.0) as u8;
//     }
//     println!("{:?} {:?}", min, max);
//
//     image::save_buffer(
//         "/Users/admin/Downloads/govno.jpg",
//         &msdf_b,
//         64,
//         64,
//         image::ColorType::Rgb8,
//     )
//     .unwrap();
// }
