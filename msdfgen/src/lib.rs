use font_kit::error::GlyphLoadingError;
use font_kit::font::Font;
use font_kit::hinting::HintingOptions;
use font_kit::outline::OutlineSink;
use pathfinder_geometry::line_segment::LineSegment2F;
use pathfinder_geometry::vector::Vector2F;
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
    pub fn from_f32(v: Vector2F, offset: Vector2F, scale: Vector2F) -> Self {
        let res = offset + v * scale;
        Self {
            x: res.x() as f64,
            y: res.y() as f64,
        }
    }
}

struct SysOutlineBuilder<'a> {
    offset: Vector2F,
    scale: Vector2F,
    curr_pos: sys::Vector2,
    shape: Pin<&'a mut sys::Shape>,
    contour: *mut sys::Contour,
}

impl OutlineSink for SysOutlineBuilder<'_> {
    fn move_to(&mut self, to: Vector2F) {
        if self.contour.is_null() || unsafe { !sys::contour_is_edges_empty(self.contour) } {
            let contour = self.shape.as_mut().addContour();
            self.contour = unsafe { Pin::get_unchecked_mut(contour) };
        }
        self.curr_pos = sys::Vector2::from_f32(to, self.offset, self.scale);
    }

    fn line_to(&mut self, to: Vector2F) {
        let endpoint = sys::Vector2::from_f32(to, self.offset, self.scale);
        if endpoint != self.curr_pos {
            unsafe { sys::contour_add_edge2(self.contour, self.curr_pos, endpoint) };
            self.curr_pos = endpoint;
        }
    }

    fn quadratic_curve_to(&mut self, ctrl: Vector2F, to: Vector2F) {
        let control = sys::Vector2::from_f32(ctrl, self.offset, self.scale);
        let to = sys::Vector2::from_f32(to, self.offset, self.scale);
        unsafe { sys::contour_add_edge3(self.contour, self.curr_pos, control, to) };
        self.curr_pos = to;
    }

    fn cubic_curve_to(&mut self, ctrl: LineSegment2F, to: Vector2F) {
        let control1 = sys::Vector2::from_f32(ctrl.from(), self.offset, self.scale);
        let control2 = sys::Vector2::from_f32(ctrl.to(), self.offset, self.scale);
        let to = sys::Vector2::from_f32(to, self.offset, self.scale);
        unsafe { sys::contour_add_edge4(self.contour, self.curr_pos, control1, control2, to) };
        self.curr_pos = to;
    }

    fn close(&mut self) {}
}

/// Generates an image of R8G8B8A8 layout with width and height of `size`.
pub fn generate_msdf(
    font: &Font,
    glyph_id: u32,
    size: u32,
    px_range: f32,
    output: &mut [u8],
) -> Result<(), GlyphLoadingError> {
    let size_f = size as f32;
    let rect = font.typographic_bounds(glyph_id)?;
    let scaled_rect = rect * (size_f / font.metrics().units_per_em as f32);

    let fit_scale = size_f / scaled_rect.width().max(scaled_rect.height());
    let total_scale = size_f * fit_scale / font.metrics().units_per_em as f32;

    let mut shape = sys::create_shape();
    let mut builder = SysOutlineBuilder {
        offset: Vector2F::new(-scaled_rect.min_x(), -scaled_rect.min_y() + size_f),
        scale: Vector2F::new(total_scale, -total_scale),
        curr_pos: Default::default(),
        shape: shape.pin_mut(),
        contour: ptr::null_mut(),
    };
    dbg!(scaled_rect, builder.scale, builder.offset);
    font.outline(glyph_id, HintingOptions::None, &mut builder)?;

    sys::shape_check_last_contour(shape.pin_mut());
    shape.pin_mut().normalize();
    sys::edgeColoringSimple(shape.pin_mut(), 3.0, 0);

    let data_f_len = (size * size * 3) as usize;
    let mut data_f = Vec::<f32>::with_capacity(data_f_len);
    unsafe {
        sys::generateMSDF(data_f.as_mut_ptr(), size, size, &shape, px_range as f64);
        data_f.set_len(data_f_len);
    };

    for (b, f) in output.chunks_exact_mut(4).zip(data_f.chunks_exact_mut(3)) {
        for (b, f) in b[0..3].iter_mut().zip(f) {
            *b = 255 - (*f * 256.0).clamp(0.0, 255.0) as u8;
        }
        b[3] = 255;
    }

    Ok(())
}

// #[test]
// fn gov() {
//     let f = std::fs::read("/Users/admin/Downloads/Romanesco-Regular.ttf").unwrap();
//     let font = Font::from_bytes(std::sync::Arc::new(f), 0).unwrap();
//     let msdf = generate_msdf(&font, font.glyph_for_char('A').unwrap(), 64, 4.0).unwrap();
//
//     let mut msdf_b = vec![0_u8; 64 * 64 * 3];
//     let min = msdf.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
//     let max = msdf.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
//
//     for (f, u) in msdf.iter().zip(&mut msdf_b) {
//         *u = 255 - (*f * 256.0).clamp(0.0, 255.0) as u8;
//     }
//     println!("{:?} {:?}", min, max);
//
//     image::save_buffer(
//         "/Users/admin/Downloads/govno.png",
//         &msdf_b,
//         64,
//         64,
//         image::ColorType::Rgb8,
//     )
//     .unwrap();
// }
