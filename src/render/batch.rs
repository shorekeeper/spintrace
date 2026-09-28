//! Draw list construction.
//!
//! The interface emits indexed quads into one vertex stream and one index
//! stream. Commands split only when the sampled texture or the clip rectangle
//! changes, so adjacent geometry with the same state is submitted together.
//!
//! Coordinates are framebuffer pixels with the origin at the top left. Solid
//! primitives bind the white texture explicitly, which keeps their output
//! independent of the texture used by the preceding text or image command.

use crate::render::texture::TextureId;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    pub fn from_min_max(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
    }

    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    pub fn is_empty(&self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    pub fn inset(&self, amount: f32) -> Rect {
        Rect {
            x: self.x + amount,
            y: self.y + amount,
            w: (self.w - amount * 2.0).max(0.0),
            h: (self.h - amount * 2.0).max(0.0),
        }
    }

    pub fn intersect(&self, other: &Rect) -> Rect {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        Rect { x: x0, y: y0, w: (x1 - x0).max(0.0), h: (y1 - y0).max(0.0) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color(pub [u8; 4]);

impl Color {
    pub const TRANSPARENT: Color = Color([0, 0, 0, 0]);

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
        Color([r, g, b, a])
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
        Color([r, g, b, 255])
    }

    pub const fn hex(value: u32) -> Color {
        Color([
            ((value >> 16) & 0xFF) as u8,
            ((value >> 8) & 0xFF) as u8,
            (value & 0xFF) as u8,
            255,
        ])
    }

    pub fn with_alpha(self, alpha: f32) -> Color {
        let mut color = self;
        color.0[3] = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
        color
    }

    pub fn lerp(self, other: Color, amount: f32) -> Color {
        let amount = amount.clamp(0.0, 1.0);
        let mut out = [0u8; 4];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (self.0[i] as f32
                + (other.0[i] as f32 - self.0[i] as f32) * amount) as u8;
        }
        Color(out)
    }
}

/// Texture interpretation retained in the vertex stream.
///
/// Texture views establish the sampling semantics used by the current
/// pipeline. The value remains part of the vertex ABI so later data views can
/// add specialized fragment paths without changing the draw list layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Solid = 0,
    Alpha = 1,
    Rgba = 2,
    Luma = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
    pub mode: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct DrawCmd {
    pub index_offset: u32,
    pub index_count: u32,
    pub texture: TextureId,
    /// Scissor rectangle as x0, y0, x1, y1 in integer pixels.
    pub clip: [i32; 4],
}

pub struct DrawList {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub commands: Vec<DrawCmd>,
    clip_stack: Vec<Rect>,
    current_clip: Rect,
    current_texture: TextureId,
    solid_texture: TextureId,
    open_index: u32,
    surface: Rect,
}

impl DrawList {
    pub fn new() -> DrawList {
        DrawList {
            vertices: Vec::with_capacity(4096),
            indices: Vec::with_capacity(6144),
            commands: Vec::with_capacity(64),
            clip_stack: Vec::with_capacity(16),
            current_clip: Rect::default(),
            current_texture: TextureId(0),
            solid_texture: TextureId(0),
            open_index: 0,
            surface: Rect::default(),
        }
    }

    /// Resets all transient state for one framebuffer.
    pub fn begin(&mut self, width: f32, height: f32, white: TextureId) {
        self.vertices.clear();
        self.indices.clear();
        self.commands.clear();
        self.clip_stack.clear();
        self.surface = Rect::new(0.0, 0.0, width, height);
        self.current_clip = self.surface;
        self.current_texture = white;
        self.solid_texture = white;
        self.open_index = 0;
    }

    /// Closes the trailing command before the list is submitted.
    pub fn end(&mut self) {
        self.flush_command();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn flush_command(&mut self) {
        let count = self.indices.len() as u32 - self.open_index;
        if count == 0 {
            return;
        }
        self.commands.push(DrawCmd {
            index_offset: self.open_index,
            index_count: count,
            texture: self.current_texture,
            clip: [
                self.current_clip.x.floor() as i32,
                self.current_clip.y.floor() as i32,
                self.current_clip.right().ceil() as i32,
                self.current_clip.bottom().ceil() as i32,
            ],
        });
        self.open_index = self.indices.len() as u32;
    }

    pub fn push_clip(&mut self, rect: Rect) {
        self.flush_command();
        self.clip_stack.push(self.current_clip);
        self.current_clip = self.current_clip.intersect(&rect);
    }

    pub fn pop_clip(&mut self) {
        self.flush_command();
        self.current_clip = self.clip_stack.pop().unwrap_or(self.surface);
    }

    pub fn clip(&self) -> Rect {
        self.current_clip
    }

    pub fn set_texture(&mut self, texture: TextureId) {
        if texture != self.current_texture {
            self.flush_command();
            self.current_texture = texture;
        }
    }

    fn use_solid_texture(&mut self) {
        self.set_texture(self.solid_texture);
    }

    fn quad(
        &mut self,
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        uv0: [f32; 2],
        uv1: [f32; 2],
        color: Color,
        mode: Mode,
    ) {
        let base = self.vertices.len() as u32;
        let color = color.0;
        let mode = mode as u32;
        self.vertices.push(Vertex { pos: p0, uv: [uv0[0], uv0[1]], color, mode });
        self.vertices.push(Vertex { pos: p1, uv: [uv1[0], uv0[1]], color, mode });
        self.vertices.push(Vertex { pos: p2, uv: [uv1[0], uv1[1]], color, mode });
        self.vertices.push(Vertex { pos: p3, uv: [uv0[0], uv1[1]], color, mode });
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        if rect.is_empty() || color.0[3] == 0 {
            return;
        }
        self.use_solid_texture();
        self.quad(
            [rect.x, rect.y],
            [rect.right(), rect.y],
            [rect.right(), rect.bottom()],
            [rect.x, rect.bottom()],
            [0.0, 0.0],
            [1.0, 1.0],
            color,
            Mode::Solid,
        );
    }

    pub fn stroke_rect(&mut self, rect: Rect, thickness: f32, color: Color) {
        if rect.is_empty() || thickness <= 0.0 {
            return;
        }
        let thickness = thickness.min(rect.w * 0.5).min(rect.h * 0.5);
        self.fill_rect(Rect::new(rect.x, rect.y, rect.w, thickness), color);
        self.fill_rect(
            Rect::new(rect.x, rect.bottom() - thickness, rect.w, thickness),
            color,
        );
        self.fill_rect(
            Rect::new(rect.x, rect.y + thickness, thickness, rect.h - thickness * 2.0),
            color,
        );
        self.fill_rect(
            Rect::new(
                rect.right() - thickness,
                rect.y + thickness,
                thickness,
                rect.h - thickness * 2.0,
            ),
            color,
        );
    }

    pub fn hline(
        &mut self,
        x0: f32,
        x1: f32,
        y: f32,
        thickness: f32,
        color: Color,
    ) {
        self.fill_rect(
            Rect::new(x0.round(), y.round(), (x1 - x0).round(), thickness),
            color,
        );
    }

    pub fn vline(
        &mut self,
        x: f32,
        y0: f32,
        y1: f32,
        thickness: f32,
        color: Color,
    ) {
        self.fill_rect(
            Rect::new(x.round(), y0.round(), thickness, (y1 - y0).round()),
            color,
        );
    }

    pub fn line(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        thickness: f32,
        color: Color,
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= 0.0001 || thickness <= 0.0 {
            return;
        }

        self.use_solid_texture();
        let nx = -dy / length * thickness * 0.5;
        let ny = dx / length * thickness * 0.5;
        self.quad(
            [x0 + nx, y0 + ny],
            [x1 + nx, y1 + ny],
            [x1 - nx, y1 - ny],
            [x0 - nx, y0 - ny],
            [0.0, 0.0],
            [1.0, 1.0],
            color,
            Mode::Solid,
        );
    }

    pub fn image(
        &mut self,
        rect: Rect,
        texture: TextureId,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
        tint: Color,
        mode: Mode,
    ) {
        if rect.is_empty() || tint.0[3] == 0 {
            return;
        }
        self.set_texture(texture);
        self.quad(
            [rect.x, rect.y],
            [rect.right(), rect.y],
            [rect.right(), rect.bottom()],
            [rect.x, rect.bottom()],
            uv_min,
            uv_max,
            tint,
            mode,
        );
    }

    pub fn gradient_v(&mut self, rect: Rect, top: Color, bottom: Color) {
        if rect.is_empty() {
            return;
        }

        self.use_solid_texture();
        let base = self.vertices.len() as u32;
        let mode = Mode::Solid as u32;
        self.vertices.push(Vertex {
            pos: [rect.x, rect.y],
            uv: [0.0, 0.0],
            color: top.0,
            mode,
        });
        self.vertices.push(Vertex {
            pos: [rect.right(), rect.y],
            uv: [1.0, 0.0],
            color: top.0,
            mode,
        });
        self.vertices.push(Vertex {
            pos: [rect.right(), rect.bottom()],
            uv: [1.0, 1.0],
            color: bottom.0,
            mode,
        });
        self.vertices.push(Vertex {
            pos: [rect.x, rect.bottom()],
            uv: [0.0, 1.0],
            color: bottom.0,
            mode,
        });
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

impl Default for DrawList {
    fn default() -> Self {
        DrawList::new()
    }
}