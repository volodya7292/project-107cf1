use std::pin::Pin;
use std::ptr;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[cxx::bridge]
mod sys {
    #[derive(Default, Copy, Clone, PartialEq)]
    pub struct Vector2 {
        pub x: f64,
        pub y: f64,
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
        pub unsafe fn generateMSDF(
            output: *mut f32,
            width: u32,
            height: u32,
            offset: Vector2,
            shape: &Shape,
            range: f64,
        );
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

/// Generates an image of R8G8B8A8 layout with width and height of `size`.
pub fn generate_msdf(
    glyph: rusttype::Glyph,
    size: u32,
    px_range: u32,
    output: &mut [u8],
) -> Result<(), &'static str> {
    let scaled = glyph.scaled(rusttype::Scale::uniform(1.0));
    let bounds = scaled
        .exact_bounding_box()
        .ok_or("Could not extract glyph bounding box")?;

    // FIXME: adjust based on offset to account for px_range
    // FIXME: invert y_offset

    let px_range = px_range as f32;
    let render_scale = (size as f32 - px_range * 2.0) / bounds.width().max(bounds.height());
    let offset = sys::Vector2 {
        x: px_range as f64,
        y: px_range as f64,
    };

    let scaled = scaled
        .into_unscaled()
        .scaled(rusttype::Scale::uniform(render_scale));
    let positioned = scaled.positioned(rusttype::Point { x: 0.0, y: 0.0 });

    let mut shape = sys::create_shape();
    let mut builder = SysOutlineBuilder {
        position: Default::default(),
        shape: shape.pin_mut(),
        contour: ptr::null_mut(),
    };

    if !positioned.build_outline(&mut builder) {
        return Err("Failed to build glyph outline");
    }

    sys::shape_check_last_contour(shape.pin_mut());
    shape.pin_mut().normalize();
    sys::edgeColoringSimple(shape.pin_mut(), 3.0, 0);

    let data_f_len = (size * size * 3) as usize;
    let mut data_f = Vec::<f32>::with_capacity(data_f_len);
    unsafe {
        sys::generateMSDF(data_f.as_mut_ptr(), size, size, offset, &shape, px_range as f64);
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

pub struct ReverseTransform {
    pub offset_x: f32,
    pub offset_y: f32,
    pub scale: f32,
}

/// In a MSDF image the glyph is fit to cover maximum width/height of the image.
/// Use this transform to reverse the glyph to its original size and position.
pub fn glyph_reverse_transform(
    glyph: rusttype::Glyph,
    size: u32,
    px_range: u32,
) -> Result<ReverseTransform, &'static str> {
    let scale = rusttype::Scale::uniform(1.0);
    let scaled = glyph.scaled(scale);
    let bounds = scaled
        .exact_bounding_box()
        .ok_or("Could not extract glyph bounding box")?;

    let size = size as f32;
    let px_range = px_range as f32;
    let width = bounds.width();
    let height = bounds.height();

    let px_downscale = 1.0 - px_range * 2.0 / size;
    let reverse_scale = width.max(height) / px_downscale;

    // Calculate free vertical space to move glyph coordinates origin from top-left to bottom-left
    let free_height = (1.0 - height / width).max(0.0) * reverse_scale;

    let px_range_offset = px_range / size * reverse_scale;

    Ok(ReverseTransform {
        offset_x: bounds.min.x - px_range_offset,
        offset_y: -bounds.max.y - px_range_offset - free_height,
        scale: reverse_scale,
    })
}

// #[test]
// fn gov() {
//     let f = std::fs::read("/Users/admin/Downloads/TimesNewRoman.ttf").unwrap();
//     let font = rusttype::Font::try_from_bytes(&f).unwrap();
//     let g = font.glyph('N');
//     let mut output = vec![0_u8; 64 * 64 * 4];
//     generate_msdf(g, 64, 4, &mut output).unwrap();
//
//     image::save_buffer(
//         "/Users/admin/Downloads/govno.png",
//         &output,
//         64,
//         64,
//         image::ColorType::Rgba8,
//     )
//     .unwrap();
// }
