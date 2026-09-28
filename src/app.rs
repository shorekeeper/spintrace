//! Application lifetime, persistent settings and frame assembly.
//!
//! The INI file is loaded before the window, Vulkan instance and font atlas are
//! created. Resource construction therefore observes device selection, buffer
//! sizes, validation, presentation and font settings from the first allocation.
//!
//! Native messages are translated before frame construction and accumulated in
//! the GUI input snapshot. The widget tree is then declared, solved and drawn.
//! Application data views occupy rectangles reserved by the tree and are
//! emitted before the GUI top layer.
//!
//! Window commands are applied after submission. A system move loop is modal
//! and can outlive the press that started it, so the GUI releases pointer
//! ownership before native movement begins.
//!
//! Mutable settings and the edited sequence are written on a clean shutdown.
//! The retained INI document preserves unknown sections, unknown keys and the
//! documentation generated with a new file.

use std::time::{Duration, Instant};

use crate::config::{ConfigFile, DEFAULT_CONFIG_PATH};
use crate::core::Result;
use crate::font::FontSystem;
use crate::gui::theme::Theme;
use crate::gui::{Frame as GuiFrame, Ui};
use crate::platform::{Event, Window};
use crate::render::{Color, DrawList, Rect, Renderer};
use crate::view::{ViewDiagnostics, ViewState};

pub fn run() -> Result<()> {
    let mut config = ConfigFile::load(DEFAULT_CONFIG_PATH)?;
    let initial = config.settings.clone();

    let mut window = Window::new(
        "Spintrace",
        initial.window.width,
        initial.window.height,
    )?;
    if initial.window.maximized {
        window.toggle_maximized();
    }

    let mut renderer = Renderer::new(&window, &initial.render)?;
    let mut fonts = FontSystem::new(
        &initial.font.ui_path,
        &initial.font.mono_path,
        initial.font.atlas_size,
        initial.font.gamma,
        &mut renderer,
    )?;

    let accent = Color::hex(initial.accent_rgb);
    let mut theme = Theme::dark(accent);
    theme.apply(accent, &initial.appearance);

    let mut ui = Ui::new(theme);
    let mut view = ViewState::new(&initial);
    let mut draw_list = DrawList::new();
    let mut top_list = DrawList::new();
    let mut events: Vec<Event> = Vec::with_capacity(64);

    let started = Instant::now();
    let mut previous = started;
    let mut frame_ms = 0.0f32;

    crate::log_info!("app", "renderer ready on {}", renderer.device_name());
    crate::log_info!("app", "interface font {}", fonts.family_name());
    crate::log_info!("app", "config {}", config.path().display());

    while window.poll_events(&mut events) {
        let frame_started = Instant::now();

        for event in events.drain(..) {
            ui.on_event(&event);
        }

        let now = Instant::now();
        let delta = now.duration_since(previous).as_secs_f32();
        previous = now;
        view.advance(delta);

        let sample_ms = delta * 1_000.0;
        frame_ms = if frame_ms <= 0.0 {
            sample_ms
        } else {
            frame_ms + (sample_ms - frame_ms) * 0.08
        };
        let fps = if frame_ms > 0.0 { 1_000.0 / frame_ms } else { 0.0 };

        let (width, height) = window.client_size();
        renderer.resize(width, height);

        let time = now.duration_since(started).as_secs_f32();
        let viewport = Rect::new(0.0, 0.0, width as f32, height as f32);
        let ui_scale = view.ui_scale() * window.dpi_scale();
        let font_px = (view.ui_font_px() * ui_scale).round();
        ui.starts_frame(viewport, ui_scale, font_px, font_px, time);

        let render_stats = renderer.stats();
        let compute_stats = renderer.compute_diagnostics();
        let device_name = renderer.device_name().to_string();
        let present_mode = renderer.active_present_mode();
        let diagnostics = ViewDiagnostics {
            device_name: &device_name,
            frame_ms,
            fps,
            surface: (width, height),
            present_mode: &present_mode,
            vsync: view.vsync(),
            render: render_stats,
            compute: compute_stats,
        };

        let actions = {
            let mut frame = GuiFrame::new(&mut ui, &mut fonts);
            view.build(&mut frame, &diagnostics, window.is_maximized())
        };

        ui.ends_frame();

        renderer.set_vsync(view.vsync());
        renderer.set_present_mode(view.present_mode());
        view.sync_compute(&mut renderer)?;

        draw_list.begin(width as f32, height as f32, renderer.white_texture());
        ui.draw_tree(&mut fonts, &mut draw_list);
        view.draw_data(&ui, &mut fonts, &mut draw_list);
        draw_list.end();

        top_list.begin(width as f32, height as f32, renderer.white_texture());
        ui.draw_top(&mut fonts, &mut top_list);

        let border = if ui.window_focused() {
            ui.theme.accent
        } else {
            ui.theme.border_strong
        };
        top_list.stroke_rect(viewport, ui.line(1.0), border);
        top_list.end();

        let visual = view.spin_visual(&ui);
        fonts.flush(&mut renderer)?;
        renderer.render(&draw_list, &top_list, visual)?;
        window.set_cursor(ui.cursor());

        if actions.close {
            window.close();
        } else {
            if actions.toggle_maximized {
                window.toggle_maximized();
            }
            if actions.minimize {
                window.minimize();
            }
            if actions.drag {
                ui.release_pointer();
                window.begin_drag();
            }
        }

        let limit = view.frame_limit();
        if limit > 0 {
            let target = Duration::from_secs_f32(1.0 / limit as f32);
            let elapsed = frame_started.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
    }

    view.store_settings(&mut config.settings);

    let normal = window.normal_size();
    config.settings.window.width = normal.0.max(640);
    config.settings.window.height = normal.1.max(480);
    config.settings.window.maximized = window.is_maximized();
    config.save()
}