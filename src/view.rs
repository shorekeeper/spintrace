//! Main application view.
//!
//! Widget declarations contain controls and reserve rectangles for data views.
//! Plot geometry is emitted after layout, between the widget tree and its top
//! layer, so popup lists and diagnostic overlays remain above application drawn
//! content.
//!
//! The view owns a retained CPU reference result. A physical parameter change
//! recompiles the pulse sequence and runs the ensemble once; ordinary animation
//! selects an already stored observation frame and performs no integration.

use std::f32::consts::TAU;

use crate::config::settings::PresentModeCfg;
use crate::config::{AppSettings, SignalViewCfg};
use crate::core::Result;
use crate::font::{FontId, FontSystem};
use crate::gui::layout::{Align, Style};
use crate::gui::{Frame, TextAlign, Ui, WindowButton};
use crate::render::{
    Color, ComputeDiagnostics, DrawList, Rect, Renderer, RenderStats,
    SpinVisualFrame,
};
use crate::sim::{
    CpuPreview, FieldEvent, RfShape, SequenceEvent, SequenceEventKind,
    SequenceKind, SequenceProgram, SignalSample, SimulationConfig,
    SimulationTrace, MAX_SEQUENCE_EVENTS,
};

const VIEW_SEQUENCE: u32 = 1;
const VIEW_MAGNETIZATION: u32 = 2;
const VIEW_SIGNAL: u32 = 3;

const SPIN_ECHO: usize = 0;
const GRADIENT_ECHO: usize = 1;
const CUSTOM_SEQUENCE: usize = 2;

const SIGNAL_COMPLEX: usize = 0;
const SIGNAL_MAGNITUDE: usize = 1;
const SIGNAL_ACQUIRED: usize = 2;

const TEXT: Color = Color::rgb(230, 234, 238);
const TEXT_DIM: Color = Color::rgb(143, 154, 165);
const BORDER: Color = Color::rgb(54, 64, 74);
const GRID: Color = Color::rgb(43, 52, 61);
const DATA_BACKGROUND: Color = Color::rgb(10, 12, 15);
const ACCENT: Color = Color::rgb(74, 158, 255);
const RF: Color = Color::rgb(229, 119, 90);
const GRADIENT: Color = Color::rgb(93, 190, 139);
const REAL: Color = Color::rgb(245, 204, 92);
const IMAGINARY: Color = Color::rgb(74, 158, 255);
const MAGNITUDE: Color = Color::rgb(93, 190, 139);

#[derive(Debug, Clone, Copy)]
pub struct ViewDiagnostics<'a> {
    pub device_name: &'a str,
    pub frame_ms: f32,
    pub fps: f32,
    pub surface: (u32, u32),
    pub present_mode: &'a str,
    pub vsync: bool,
    pub render: RenderStats,
    pub compute: ComputeDiagnostics,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WindowActions {
    pub drag: bool,
    pub toggle_maximized: bool,
    pub minimize: bool,
    pub close: bool,
}

/// Editable application state and retained CPU and GPU simulation results.
///
/// GUI values use display units. model_config performs the conversion to SI
/// units before either backend receives a configuration.
pub struct ViewState {
    sequence: usize,
    running: bool,
    spin_count: u32,
    signal_samples: u32,
    t1_ms: f32,
    t2_ms: f32,
    t1_spread_pct: f32,
    t2_spread_pct: f32,
    center_offset_hz: f32,
    offset_span_hz: f32,
    sample_width_mm: f32,
    sample_height_mm: f32,
    sample_depth_mm: f32,

    te_ms: f32,
    excitation_flip_deg: f32,
    refocus_flip_deg: f32,
    rf_phase_deg: f32,
    rf_duration_ms: f32,
    gradient_mtm: f32,
    adc_duration_ms: f32,

    time_scale: f32,
    vector_scale: f32,
    signal_view: usize,
    show_phase: bool,
    show_diagnostics: bool,

    vsync: bool,
    present_mode: usize,
    frame_limit: u32,
    ui_scale: f32,
    ui_font_px: f32,
    elapsed_s: f32,
    program: SequenceProgram,
    program_revision: u64,
    selected_event: usize,
    preview: CpuPreview,
    gpu_key: Option<(SimulationConfig, u64)>,
    candidate_gpu_key: Option<(SimulationConfig, u64)>,
    requested_gpu_key: Option<(SimulationConfig, u64)>,
    gpu_trace: Option<SimulationTrace>,
}

impl ViewState {
    pub fn new(settings: &AppSettings) -> ViewState {
        let model = settings.model;
        let program = settings.program.clone();
        let mut preview_model = model;
        preview_model.spin_count = 512;
        let preview = CpuPreview::with_program(preview_model, program.clone());

        ViewState {
            sequence: match model.sequence {
                SequenceKind::SpinEcho => SPIN_ECHO,
                SequenceKind::GradientEcho => GRADIENT_ECHO,
                SequenceKind::Custom => CUSTOM_SEQUENCE,
            },
            running: settings.running,
            spin_count: model.spin_count as u32,
            signal_samples: model.observation_count as u32,
            t1_ms: model.t1_s * 1_000.0,
            t2_ms: model.t2_s * 1_000.0,
            t1_spread_pct: model.t1_spread * 100.0,
            t2_spread_pct: model.t2_spread * 100.0,
            center_offset_hz: model.center_offset_hz,
            offset_span_hz: model.offset_span_hz,
            sample_width_mm: model.sample_extent_m[0] * 1_000.0,
            sample_height_mm: model.sample_extent_m[1] * 1_000.0,
            sample_depth_mm: model.sample_extent_m[2] * 1_000.0,

            te_ms: model.te_s * 1_000.0,
            excitation_flip_deg: model.excitation_flip_rad.to_degrees(),
            refocus_flip_deg: model.refocus_flip_rad.to_degrees(),
            rf_phase_deg: signed_degrees(model.rf_phase_rad),
            rf_duration_ms: model.rf_duration_s * 1_000.0,
            gradient_mtm: model.gradient_amplitude_t_m * 1_000.0,
            adc_duration_ms: model.adc_duration_s * 1_000.0,

            time_scale: settings.time_scale,
            vector_scale: settings.vector_scale,
            signal_view: settings.signal_view.index(),
            show_phase: settings.show_phase,
            show_diagnostics: settings.show_diagnostics,

            vsync: settings.render.vsync,
            present_mode: settings.render.present_mode.index(),
            frame_limit: settings.frame_limit,
            ui_scale: settings.ui_scale,
            ui_font_px: settings.ui_font_px,

            elapsed_s: 0.0,
            program,
            program_revision: 1,
            selected_event: 0,
            preview,
            gpu_key: None,
            candidate_gpu_key: None,
            requested_gpu_key: None,
            gpu_trace: None,
        }
    }

    pub fn advance(&mut self, seconds: f32) {
        if self.running {
            self.elapsed_s += seconds.clamp(0.0, 0.1) * self.time_scale;
        }
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn present_mode(&self) -> PresentModeCfg {
        PresentModeCfg::from_index(self.present_mode)
    }

    pub fn frame_limit(&self) -> u32 {
        self.frame_limit
    }

    pub fn ui_scale(&self) -> f32 {
        self.ui_scale
    }

    pub fn ui_font_px(&self) -> f32 {
        self.ui_font_px
    }

    /// Copies the mutable interface state into the persistent settings model.
    pub fn store_settings(&self, settings: &mut AppSettings) {
        settings.render.vsync = self.vsync;
        settings.render.present_mode = self.present_mode();
        settings.model = self.gpu_model_config().sanitized();
        settings.program = self.program.clone();

        settings.running = self.running;
        settings.time_scale = self.time_scale;
        settings.vector_scale = self.vector_scale;
        settings.signal_view = SignalViewCfg::from_index(self.signal_view);
        settings.show_phase = self.show_phase;
        settings.show_diagnostics = self.show_diagnostics;
        settings.frame_limit = self.frame_limit;
        settings.ui_scale = self.ui_scale;
        settings.ui_font_px = self.ui_font_px;
    }

    pub fn build(
        &mut self,
        frame: &mut Frame<'_>,
        diagnostics: &ViewDiagnostics<'_>,
        maximized: bool,
    ) -> WindowActions {
        let mut actions = WindowActions::default();
        let header_height = frame.ui.m(42.0);
        let panel = frame.ui.theme.panel;
        let panel_header = frame.ui.theme.panel_header;
        let border = frame.ui.theme.border;
        let gap = frame.ui.m(10.0);
        let sidebar_width = frame.ui.m(286.0);
        frame.ui.set_panel_width(286.0);

        frame.begin_frame(
            Style::row()
                .height_px(header_height)
                .padding_xy(frame.ui.m(12.0), 0.0)
                .gap(frame.ui.m(16.0))
                .align(Align::Center),
            panel_header,
            border,
        );
        frame.label_styled(
            "Spintrace",
            TEXT,
            FontId::Ui,
            TextAlign::Left,
            Style::row().height_px(header_height),
        );
        frame.label_styled(
            "Bloch spin and pulse sequence simulator",
            TEXT_DIM,
            FontId::Ui,
            TextAlign::Left,
            Style::row().shrink(1.0).height_px(header_height),
        );

        let caption = frame.caption(
            "window.caption",
            Style::row().grow(1.0).align_self(Align::Stretch),
        );
        actions.drag = caption.drag;
        actions.toggle_maximized = caption.double;

        frame.label_mono(
            if self.running { "RUNNING" } else { "PAUSED" },
            if self.running { MAGNITUDE } else { TEXT_DIM },
            TextAlign::Right,
            Style::row().width_px(frame.ui.m(72.0)).height_px(header_height),
        );
        if frame.button("Reset") {
            self.elapsed_s = 0.0;
        }
        if frame.window_button("window.minimize", WindowButton::Minimize) {
            actions.minimize = true;
        }
        let maximize = if maximized {
            WindowButton::Restore
        } else {
            WindowButton::Maximize
        };
        if frame.window_button("window.maximize", maximize) {
            actions.toggle_maximized = true;
        }
        if frame.window_button("window.close", WindowButton::Close) {
            actions.close = true;
        }
        frame.end();

        frame.begin(
            Style::row()
                .grow(1.0)
                .shrink(1.0)
                .min_h(frame.ui.m(180.0))
                .padding(gap)
                .gap(gap),
        );

        frame.begin_frame(
            Style::column()
                .width_px(sidebar_width)
                .shrink(0.0)
                .padding(frame.ui.m(6.0)),
            panel,
            border,
        );
        frame.begin_scroll(
            "controls",
            Style::column().grow(1.0).shrink(1.0).gap(frame.ui.m(8.0)),
        );

        let simulation_open = frame.begin_group("SIMULATION");
        frame.mark_active(self.running);
        if simulation_open {
            if frame.combo(
                "Sequence",
                &mut self.sequence,
                &["Spin echo", "Gradient echo", "Custom"],
            ) && self.sequence != CUSTOM_SEQUENCE {
                self.rebuild_preset();
            }
            frame.toggle("Running", &mut self.running);
            frame.slider(
                "Time scale",
                &mut self.time_scale,
                0.02,
                1.0,
                2,
                "x",
            );

            let backend = format!(
                "Vulkan {} / CPU view {}",
                diagnostics.compute.model_spins,
                self.preview.output().spin_count()
            );
            frame.readout("Backend", &backend);
        }
        frame.end_group();

        let preset_open = frame.begin_group("PRESET");
        if preset_open {
            let mut changed = false;
            frame.begin_disabled(self.sequence == CUSTOM_SEQUENCE);

            changed |= frame.slider("TE", &mut self.te_ms, 10.0, 250.0, 1, "ms");

            let rf_max = (self.te_ms * 0.20).clamp(0.05, 5.0);
            changed |= frame.slider(
                "RF duration",
                &mut self.rf_duration_ms,
                0.02,
                rf_max,
                2,
                "ms",
            );
            changed |= frame.slider(
                "Excitation flip",
                &mut self.excitation_flip_deg,
                0.0,
                180.0,
                1,
                "deg",
            );

            frame.begin_disabled(self.sequence != SPIN_ECHO);
            changed |= frame.slider(
                "Refocus flip",
                &mut self.refocus_flip_deg,
                0.0,
                360.0,
                1,
                "deg",
            );
            frame.end_disabled();

            changed |= frame.slider(
                "RF phase",
                &mut self.rf_phase_deg,
                -180.0,
                180.0,
                1,
                "deg",
            );
            changed |= frame.slider(
                "Gradient",
                &mut self.gradient_mtm,
                0.0,
                2.0,
                3,
                "mT/m",
            );

            let adc_max = (self.te_ms * 0.80).max(0.5);
            changed |= frame.slider(
                "ADC duration",
                &mut self.adc_duration_ms,
                0.2,
                adc_max,
                1,
                "ms",
            );
            frame.end_disabled();

            if changed && self.sequence != CUSTOM_SEQUENCE {
                self.rebuild_preset();
            }
        }
        frame.end_group();

        let timeline_open = frame.begin_group("TIMELINE");
        if timeline_open {
            let mut duration_ms = self.program.duration_s * 1_000.0;
            let mut echo_ms = self.program.echo_time_s * 1_000.0;
            let mut changed = false;

            changed |= frame.slider(
                "Duration",
                &mut duration_ms,
                5.0,
                500.0,
                1,
                "ms",
            );
            changed |= frame.slider(
                "Echo marker",
                &mut echo_ms,
                0.0,
                duration_ms.max(0.1),
                1,
                "ms",
            );
            frame.slider_u32(
                "Signal samples",
                &mut self.signal_samples,
                64,
                1_024,
                "",
            );

            if changed {
                self.program.duration_s = duration_ms * 0.001;
                self.program.echo_time_s = echo_ms * 0.001;
                self.program.sanitize();
                self.mark_program_changed();
            }
        }
        frame.end_group();

        let events_open = frame.begin_group("EVENTS");
        frame.mark_active(!self.program.events.is_empty());
        if events_open {
            self.edit_sequence_events(frame);
        }
        frame.end_group();

        let ensemble_open = frame.begin_group("ENSEMBLE");
        if ensemble_open {
            frame.slider_u32("Spins", &mut self.spin_count, 1_024, 1_048_576, "");
            frame.slider("T1", &mut self.t1_ms, 100.0, 3_000.0, 0, "ms");
            frame.slider(
                "T1 spread",
                &mut self.t1_spread_pct,
                0.0,
                90.0,
                0,
                "%",
            );
            frame.slider("T2", &mut self.t2_ms, 10.0, 500.0, 0, "ms");
            frame.slider(
                "T2 spread",
                &mut self.t2_spread_pct,
                0.0,
                90.0,
                0,
                "%",
            );
            frame.slider(
                "Center offset",
                &mut self.center_offset_hz,
                -250.0,
                250.0,
                1,
                "Hz",
            );
            frame.slider(
                "Offset span",
                &mut self.offset_span_hz,
                0.0,
                500.0,
                1,
                "Hz",
            );
            frame.slider(
                "Sample width",
                &mut self.sample_width_mm,
                1.0,
                300.0,
                1,
                "mm",
            );
            frame.slider(
                "Sample height",
                &mut self.sample_height_mm,
                1.0,
                300.0,
                1,
                "mm",
            );
            frame.slider(
                "Sample depth",
                &mut self.sample_depth_mm,
                1.0,
                100.0,
                1,
                "mm",
            );
        }
        frame.end_group();

        let display_open = frame.begin_group("DISPLAY");
        frame.mark_active(self.show_phase);
        if display_open {
            frame.combo(
                "Signal view",
                &mut self.signal_view,
                &["Complex", "Magnitude", "Acquired only"],
            );
            frame.slider(
                "Vector scale",
                &mut self.vector_scale,
                0.2,
                1.0,
                2,
                "",
            );
            frame.toggle("Phase histogram", &mut self.show_phase);
            frame.toggle("Diagnostics", &mut self.show_diagnostics);
        }
        frame.end_group();

        let presentation_open = frame.begin_group("PRESENTATION");
        if presentation_open {
            frame.combo(
                "Present mode",
                &mut self.present_mode,
                &PresentModeCfg::LABELS,
            );

            frame.begin_disabled(
                self.present_mode != PresentModeCfg::Auto.index(),
            );
            frame.toggle("VSync", &mut self.vsync);
            frame.end_disabled();

            frame.slider_u32(
                "Frame limit",
                &mut self.frame_limit,
                0,
                360,
                "fps",
            );
            frame.slider(
                "UI scale",
                &mut self.ui_scale,
                0.75,
                2.0,
                2,
                "x",
            );
            frame.slider(
                "Font size",
                &mut self.ui_font_px,
                10.0,
                18.0,
                0,
                "px",
            );
            frame.readout("Active mode", diagnostics.present_mode);
            frame.hint("VSync is used by Auto present mode. A frame limit of 0 is disabled.");
        }
        frame.end_group();

        frame.end_scroll();
        frame.end();

        frame.begin(
            Style::column()
                .grow(1.0)
                .shrink(1.0)
                .min_w(frame.ui.m(320.0))
                .gap(gap),
        );

        frame.begin(
            Style::row()
                .basis_percent(0.52)
                .grow(1.0)
                .shrink(1.0)
                .min_h(frame.ui.m(180.0))
                .gap(gap),
        );

        frame.begin_panel(
            "SEQUENCE",
            Style::column()
                .basis_percent(0.62)
                .grow(1.0)
                .shrink(1.0)
                .min_w(frame.ui.m(260.0)),
        );
        frame.custom(VIEW_SEQUENCE, Style::row().grow(1.0).shrink(1.0));
        frame.end_panel();

        frame.begin_panel(
            "MAGNETIZATION / PHASE",
            Style::column()
                .basis_percent(0.38)
                .grow(1.0)
                .shrink(1.0)
                .min_w(frame.ui.m(220.0)),
        );
        frame.custom(VIEW_MAGNETIZATION, Style::row().grow(1.0).shrink(1.0));
        frame.end_panel();

        frame.end();

        frame.begin_panel(
            "MR SIGNAL",
            Style::column()
                .grow(1.0)
                .shrink(1.0)
                .min_h(frame.ui.m(170.0)),
        );
        frame.custom(VIEW_SIGNAL, Style::row().grow(1.0).shrink(1.0));
        frame.end_panel();

        frame.end();
        frame.end();

        self.preview
            .update_program(self.cpu_config(), &self.program);

        if self.show_diagnostics {
            let output = self.preview.output();
            frame.ui.set_overlay(vec![
                "Spintrace diagnostics".to_string(),
                format!("device  {}", diagnostics.device_name),
                format!(
                    "frame   {:6.2} ms  {:6.1} fps",
                    diagnostics.frame_ms,
                    diagnostics.fps
                ),
                format!(
                    "surface {} x {}",
                    diagnostics.surface.0,
                    diagnostics.surface.1
                ),
                format!(
                    "present {}  vsync {}",
                    diagnostics.present_mode,
                    if diagnostics.vsync { "on" } else { "off" }
                ),
                format!(
                    "draw    {} UI  {} spin  {} vertices",
                    diagnostics.render.draw_calls,
                    diagnostics.render.spin_draw_calls,
                    diagnostics.render.vertices
                ),
                format!(
                    "visual  {} states  GPU histogram",
                    diagnostics.render.visual_states
                ),
                format!(
                    "upload  {} copies  {} bytes",
                    diagnostics.render.uploads,
                    diagnostics.render.upload_bytes
                ),
                format!(
                    "verify  {} spins  {} dispatch  {:.3e}",
                    diagnostics.compute.verification_spins,
                    diagnostics.compute.verification_dispatches,
                    diagnostics.compute.verification_max_error
                ),
                format!(
                    "signal  verification {:.3e}",
                    diagnostics.compute.signal_verification_max_error
                ),
                format!(
                    "model   {} spins  {} steps",
                    diagnostics.compute.model_spins,
                    diagnostics.compute.model_steps
                ),
                format!(
                    "signal  {} samples  {} partials",
                    diagnostics.compute.signal_samples,
                    diagnostics.compute.partials_per_sample
                ),
                format!(
                    "compute {}  {} queue",
                    if diagnostics.compute.running { "running" } else { "idle" },
                    if diagnostics.compute.separate_queue {
                        "separate"
                    } else {
                        "shared"
                    }
                ),
                format!(
                    "run     {} dispatch  {:.2} ms",
                    diagnostics.compute.signal_dispatches,
                    diagnostics.compute.run_ms
                ),
                format!(
                    "visual  {} GPU states per sample",
                    diagnostics.compute.visual_spins
                ),
                format!(
                    "view    {} spins  {} samples",
                    output.spin_count(),
                    output.samples().len()
                ),
                format!(
                    "seq     {} events  revision {}",
                    self.program.events.len(),
                    self.program_revision
                ),
            ]);
        }

        actions
    }

    pub fn draw_data(&self, ui: &Ui, fonts: &mut FontSystem, list: &mut DrawList) {
        if let Some(rect) = ui.custom_rect(VIEW_SEQUENCE) {
            self.draw_sequence(list, fonts, rect);
        }
        if let Some(rect) = ui.custom_rect(VIEW_MAGNETIZATION) {
            self.draw_magnetization(list, fonts, rect);
        }
        if let Some(rect) = ui.custom_rect(VIEW_SIGNAL) {
            self.draw_signal(list, fonts, rect);
        }
    }

    fn rebuild_preset(&mut self) {
        let config = self.model_config(512);
        self.program = SequenceProgram::preset(&config);
        self.selected_event =
            self.selected_event.min(self.program.events.len().saturating_sub(1));
        self.bump_program_revision();
    }

    fn mark_program_changed(&mut self) {
        self.sequence = CUSTOM_SEQUENCE;
        self.bump_program_revision();
    }

    fn bump_program_revision(&mut self) {
        self.program_revision = self.program_revision.wrapping_add(1);
        if self.program_revision == 0 {
            self.program_revision = 1;
        }
    }

    /// Declares the editor for one selected event and the list operations.
    ///
    /// The event is copied while controls are active and written back only after
    /// the widget calls finish. This avoids holding a mutable vector element
    /// across calls that also need mutable access to the view.
    fn edit_sequence_events(&mut self, frame: &mut Frame<'_>) {
        let count = format!(
            "{} / {}",
            self.program.events.len(),
            MAX_SEQUENCE_EVENTS
        );
        frame.readout("Count", &count);

        if !self.program.events.is_empty() {
            self.selected_event =
                self.selected_event.min(self.program.events.len() - 1);

            let labels: Vec<String> = self
                .program
                .events
                .iter()
                .enumerate()
                .map(|(index, event)| {
                    format!(
                        "{:02}  {}  {:.2} ms",
                        index + 1,
                        event.kind.label(),
                        event.start_s * 1_000.0
                    )
                })
                .collect();
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            frame.combo("Event", &mut self.selected_event, &labels);

            let index = self.selected_event.min(self.program.events.len() - 1);
            let mut event = self.program.events[index];
            let mut changed = false;
            let mut kind = event.kind.index();

            if frame.combo("Type", &mut kind, &SequenceEventKind::LABELS) {
                event = event.converted(SequenceEventKind::from_index(kind));
                changed = true;
            }

            let total_ms = self.program.duration_s * 1_000.0;
            let mut start_ms = event.start_s * 1_000.0;
            let mut duration_ms = event.duration_s * 1_000.0;
            changed |= frame.slider(
                "Start",
                &mut start_ms,
                0.0,
                (total_ms - 0.001).max(0.0),
                3,
                "ms",
            );
            changed |= frame.slider(
                "Event duration",
                &mut duration_ms,
                0.001,
                (total_ms - start_ms).max(0.001),
                3,
                "ms",
            );

            event.start_s = start_ms * 0.001;
            event.duration_s = duration_ms * 0.001;

            match event.kind {
                SequenceEventKind::Rf => {
                    let mut shape = event.rf_shape.index();
                    if frame.combo("RF shape", &mut shape, &RfShape::LABELS) {
                        event.rf_shape = RfShape::from_index(shape);
                        changed = true;
                    }

                    let mut flip_deg = event.rf_flip_rad.to_degrees();
                    let mut phase_deg = signed_degrees(event.rf_phase_rad);
                    changed |= frame.slider(
                        "Flip",
                        &mut flip_deg,
                        0.0,
                        360.0,
                        1,
                        "deg",
                    );
                    changed |= frame.slider(
                        "Phase",
                        &mut phase_deg,
                        -180.0,
                        180.0,
                        1,
                        "deg",
                    );

                    frame.begin_disabled(event.rf_shape == RfShape::Rectangular);
                    changed |= frame.slider_u32(
                        "Waveform samples",
                        &mut event.rf_samples,
                        2,
                        256,
                        "",
                    );
                    frame.end_disabled();

                    event.rf_flip_rad = flip_deg.to_radians();
                    event.rf_phase_rad = phase_deg.to_radians();
                }

                SequenceEventKind::Gradient => {
                    let mut gx = event.gradient_t_m[0] * 1_000.0;
                    let mut gy = event.gradient_t_m[1] * 1_000.0;
                    let mut gz = event.gradient_t_m[2] * 1_000.0;

                    changed |= frame.slider("Gx", &mut gx, -10.0, 10.0, 3, "mT/m");
                    changed |= frame.slider("Gy", &mut gy, -10.0, 10.0, 3, "mT/m");
                    changed |= frame.slider("Gz", &mut gz, -10.0, 10.0, 3, "mT/m");

                    event.gradient_t_m = [gx * 0.001, gy * 0.001, gz * 0.001];
                }

                SequenceEventKind::Adc => {
                    let center = event.center_s() * 1_000.0;
                    frame.readout("Center", &format!("{:.3} ms", center));
                }
            }

            if changed {
                event.sanitize(self.program.duration_s);
                self.program.events[index] = event;
                self.mark_program_changed();
            }

            let gap = frame.ui.m(frame.ui.theme.gap);
            frame.begin(Style::row().gap(gap));

            if frame.button("Up") && index > 0 {
                self.program.events.swap(index, index - 1);
                self.selected_event = index - 1;
                self.mark_program_changed();
            }
            if frame.button("Down") && index + 1 < self.program.events.len() {
                self.program.events.swap(index, index + 1);
                self.selected_event = index + 1;
                self.mark_program_changed();
            }
            if frame.button("Remove") {
                self.program.events.remove(index);
                self.selected_event =
                    self.selected_event.min(self.program.events.len().saturating_sub(1));
                self.mark_program_changed();
            }
            frame.end();
        } else {
            frame.hint("The sequence contains no events.");
        }

        let full = self.program.events.len() >= MAX_SEQUENCE_EVENTS;
        frame.begin_disabled(full);
        let gap = frame.ui.m(frame.ui.theme.gap);
        frame.begin(Style::row().gap(gap));

        let add_rf = frame.button("Add RF");
        let add_gradient = frame.button("Add gradient");
        let add_adc = frame.button("Add ADC");

        frame.end();
        frame.end_disabled();

        if add_rf {
            let duration = (self.program.duration_s * 0.025).clamp(0.1e-3, 2.0e-3);
            let center = (self.program.echo_time_s * 0.5)
                .clamp(duration * 0.5, self.program.duration_s - duration * 0.5);
            self.program.events.push(SequenceEvent::rf(
                center,
                duration,
                std::f32::consts::PI * 0.5,
                0.0,
                RfShape::Rectangular,
                32,
            ));
            self.selected_event = self.program.events.len() - 1;
            self.mark_program_changed();
        }

        if add_gradient {
            let duration = (self.program.duration_s * 0.10).clamp(0.2e-3, 10.0e-3);
            let start = (self.program.echo_time_s * 0.35)
                .clamp(0.0, self.program.duration_s - duration);
            self.program.events.push(SequenceEvent::gradient(
                start,
                duration,
                [0.2e-3, 0.0, 0.0],
            ));
            self.selected_event = self.program.events.len() - 1;
            self.mark_program_changed();
        }

        if add_adc {
            let duration = (self.program.duration_s * 0.15).clamp(0.5e-3, 20.0e-3);
            let start = (self.program.echo_time_s - duration * 0.5)
                .clamp(0.0, self.program.duration_s - duration);
            self.program
                .events
                .push(SequenceEvent::adc(start, duration));
            self.selected_event = self.program.events.len() - 1;
            self.mark_program_changed();
        }
    }

    /// Returns the selected state snapshot and its two destination rectangles.
    ///
    /// The returned state slice is uploaded into the current frame slot and is
    /// consumed by the direct Vulkan graphics and histogram pipelines.
    pub fn spin_visual<'a>(&'a self, ui: &Ui) -> Option<SpinVisualFrame<'a>> {
        let rect = ui.custom_rect(VIEW_MAGNETIZATION)?;
        if rect.w < 120.0 || rect.h < 110.0 {
            return None;
        }

        let time_s = self.elapsed_time_s();
        let states = match self.current_gpu_trace() {
            Some(trace) => trace.states_at(time_s),
            None => self.preview.output().states_at(time_s),
        };
        if states.is_empty() {
            return None;
        }

        let histogram_height =
            if self.show_phase { (rect.h * 0.27).max(42.0) } else { 0.0 };
        let vectors = Rect::new(
            rect.x + 8.0,
            rect.y + 8.0,
            rect.w - 16.0,
            rect.h - histogram_height - 15.0,
        );
        let histogram = self.show_phase.then(|| {
            Rect::new(
                rect.x + 8.0,
                rect.bottom() - histogram_height,
                rect.w - 16.0,
                histogram_height - 8.0,
            )
        });

        Some(SpinVisualFrame {
            states,
            vectors,
            histogram,
            vector_scale: self.vector_scale,
        })
    }

    fn model_config(&self, spin_count: usize) -> SimulationConfig {
        SimulationConfig {
            sequence: match self.sequence {
                SPIN_ECHO => SequenceKind::SpinEcho,
                GRADIENT_ECHO => SequenceKind::GradientEcho,
                _ => SequenceKind::Custom,
            },
            spin_count,
            observation_count: self.signal_samples as usize,
            t1_s: self.t1_ms * 0.001,
            t2_s: self.t2_ms * 0.001,
            t1_spread: self.t1_spread_pct * 0.01,
            t2_spread: self.t2_spread_pct * 0.01,
            center_offset_hz: self.center_offset_hz,
            offset_span_hz: self.offset_span_hz,
            te_s: if self.sequence == CUSTOM_SEQUENCE {
                self.program.echo_time_s
            } else {
                self.te_ms * 0.001
            },
            excitation_flip_rad: self.excitation_flip_deg.to_radians(),
            refocus_flip_rad: self.refocus_flip_deg.to_radians(),
            rf_phase_rad: self.rf_phase_deg.to_radians(),
            rf_duration_s: self.rf_duration_ms * 0.001,
            gradient_amplitude_t_m: self.gradient_mtm * 0.001,
            adc_duration_s: self.adc_duration_ms * 0.001,
            sample_extent_m: [
                self.sample_width_mm * 0.001,
                self.sample_height_mm * 0.001,
                self.sample_depth_mm * 0.001,
            ],
            ..SimulationConfig::default()
        }
    }

    fn cpu_config(&self) -> SimulationConfig {
        self.model_config(512)
    }

    fn gpu_model_config(&self) -> SimulationConfig {
        self.model_config(self.spin_count as usize)
    }

    /// Polls the active GPU model and submits the newest stable configuration.
    ///
    /// A configuration must be present on two consecutive frames before it is
    /// submitted. While an older request is running, further changes collapse
    /// into one latest candidate rather than a queue of obsolete simulations.
    pub fn sync_compute(&mut self, renderer: &mut Renderer) -> Result<bool> {
        let desired = self.gpu_model_config().sanitized();
        let key = (desired, self.program_revision);
        let mut changed = false;

        if let Some(trace) = renderer.poll_simulation()? {
            self.requested_gpu_key = None;
            if trace.config == desired
                && trace.sequence_revision == self.program_revision
            {
                self.gpu_key = Some((trace.config, trace.sequence_revision));
                self.gpu_trace = Some(trace);
                changed = true;
            }
        }

        if self.gpu_key == Some(key) || self.requested_gpu_key == Some(key) {
            self.candidate_gpu_key = None;
            return Ok(changed);
        }

        if self.candidate_gpu_key != Some(key) {
            self.candidate_gpu_key = Some(key);
            return Ok(changed);
        }

        if self.requested_gpu_key.is_none()
            && renderer.request_simulation(
                desired,
                &self.program,
                self.program_revision,
            )?
        {
            self.requested_gpu_key = Some(key);
            self.candidate_gpu_key = None;
        }

        Ok(changed)
    }

    fn current_gpu_trace(&self) -> Option<&SimulationTrace> {
        let key = (self.gpu_model_config().sanitized(), self.program_revision);
        if self.gpu_key == Some(key) {
            self.gpu_trace.as_ref()
        } else {
            None
        }
    }

    fn elapsed_time_s(&self) -> f32 {
        self.elapsed_s.rem_euclid(self.preview.output().duration_s().max(1.0e-6))
    }

    fn draw_sequence(&self, list: &mut DrawList, fonts: &mut FontSystem, rect: Rect) {
        if rect.w < 160.0 || rect.h < 120.0 {
            return;
        }

        list.fill_rect(rect, DATA_BACKGROUND);
        list.push_clip(rect);

        let output = self.preview.output();
        let sequence = output.sequence();
        let plot = Rect::new(rect.x + 44.0, rect.y + 14.0, rect.w - 56.0, rect.h - 27.0);
        let labels = ["RF", "Gx", "Gy", "Gz", "ADC"];
        let row_height = plot.h / labels.len() as f32;

        for (index, label) in labels.iter().enumerate() {
            let center = plot.y + (index as f32 + 0.5) * row_height;
            list.hline(plot.x, plot.right(), center, 1.0, BORDER);
            fonts.draw_text(
                list,
                rect.x + 9.0,
                center + 4.0,
                label,
                FontId::Mono,
                10.0,
                TEXT_DIM,
            );
        }

        let mut rf_max = 0.0f32;
        let mut gradient_max = [0.0f32; 3];
        for event in &sequence.events {
            rf_max = rf_max
                .max(event.b1_t[0].abs())
                .max(event.b1_t[1].abs());
            for axis in 0..3 {
                gradient_max[axis] =
                    gradient_max[axis].max(event.gradient_t_m[axis].abs());
            }
        }

        for event in &sequence.events {
            let rf = if event.b1_t[0].abs() >= event.b1_t[1].abs() {
                event.b1_t[0]
            } else {
                event.b1_t[1]
            };
            if rf != 0.0 {
                draw_event(
                    list,
                    plot,
                    row_height * 0.5,
                    event,
                    rf / rf_max.max(1.0e-12),
                    RF,
                    sequence.duration_s,
                );
            }

            for axis in 0..3 {
                let value = event.gradient_t_m[axis];
                if value != 0.0 {
                    draw_event(
                        list,
                        plot,
                        row_height * (axis as f32 + 1.5),
                        event,
                        value / gradient_max[axis].max(1.0e-12),
                        if axis == 1 { ACCENT } else { GRADIENT },
                        sequence.duration_s,
                    );
                }
            }
        }

        let adc_center = plot.y + row_height * 4.5;
        for adc in &sequence.adc_blocks {
            let x0 = time_x(plot, adc.start_s, sequence.duration_s);
            let x1 = time_x(plot, adc.end_s, sequence.duration_s);
            list.fill_rect(
                Rect::from_min_max(x0, adc_center - 7.0, x1.max(x0 + 1.0), adc_center + 7.0),
                REAL,
            );
        }

        let echo_x = time_x(plot, sequence.echo_time_s, sequence.duration_s);
        list.vline(echo_x, plot.y, plot.bottom(), 1.0, ACCENT.with_alpha(0.65));
        fonts.draw_text(
            list,
            echo_x + 5.0,
            plot.y + row_height * 3.4,
            "TE",
            FontId::Mono,
            9.0,
            ACCENT,
        );

        let progress_x = time_x(plot, self.elapsed_time_s(), sequence.duration_s);
        list.vline(progress_x, plot.y, plot.bottom(), 1.0, TEXT.with_alpha(0.70));

        fonts.draw_text(
            list,
            plot.x,
            plot.bottom() + 12.0,
            match self.sequence {
                SPIN_ECHO => "Spin echo",
                GRADIENT_ECHO => "Gradient echo",
                _ => "Custom",
            },
            FontId::Ui,
            10.0,
            TEXT_DIM,
        );
        list.pop_clip();
    }

    fn draw_magnetization(
        &self,
        list: &mut DrawList,
        fonts: &mut FontSystem,
        rect: Rect,
    ) {
        if rect.w < 120.0 || rect.h < 110.0 {
            return;
        }

        list.fill_rect(rect, DATA_BACKGROUND);
        if self.show_phase {
            let histogram_height = (rect.h * 0.27).max(42.0);
            let histogram = Rect::new(
                rect.x + 8.0,
                rect.bottom() - histogram_height,
                rect.w - 16.0,
                histogram_height - 8.0,
            );
            list.hline(
                histogram.x,
                histogram.right(),
                histogram.bottom(),
                1.0,
                BORDER,
            );
            fonts.draw_text(
                list,
                histogram.x,
                histogram.y + 10.0,
                "phase",
                FontId::Ui,
                9.0,
                TEXT_DIM,
            );
        }
    }

    fn draw_signal(&self, list: &mut DrawList, fonts: &mut FontSystem, rect: Rect) {
        if rect.w < 180.0 || rect.h < 110.0 {
            return;
        }

        list.fill_rect(rect, DATA_BACKGROUND);
        list.push_clip(rect);

        let output = self.preview.output();
        let samples = match self.current_gpu_trace() {
            Some(trace) => trace.samples.as_slice(),
            None => output.samples(),
        };
        let plot = Rect::new(rect.x + 45.0, rect.y + 17.0, rect.w - 57.0, rect.h - 42.0);

        if let Some(first) = samples.iter().position(|sample| sample.acquired) {
            if let Some(last) = samples.iter().rposition(|sample| sample.acquired) {
                let x0 = time_x(plot, samples[first].time_s, output.duration_s());
                let x1 = time_x(plot, samples[last].time_s, output.duration_s());
                list.fill_rect(
                    Rect::from_min_max(x0, plot.y, x1.max(x0 + 1.0), plot.bottom()),
                    REAL.with_alpha(0.055),
                );
            }
        }

        for index in 0..=4 {
            let y = plot.y + plot.h * index as f32 / 4.0;
            list.hline(plot.x, plot.right(), y, 1.0, GRID);
        }
        for index in 0..=8 {
            let x = plot.x + plot.w * index as f32 / 8.0;
            list.vline(x, plot.y, plot.bottom(), 1.0, GRID);
        }

        let center = plot.y + plot.h * 0.5;
        let scale = plot.h * 0.42;
        list.hline(plot.x, plot.right(), center, 1.0, BORDER);

        let acquired_only = self.signal_view == SIGNAL_ACQUIRED;
        let draw_complex = self.signal_view != SIGNAL_MAGNITUDE;
        let mut previous_real: Option<(f32, f32)> = None;
        let mut previous_imaginary: Option<(f32, f32)> = None;
        let mut previous_magnitude: Option<(f32, f32)> = None;

        for sample in samples {
            if acquired_only && !sample.acquired {
                previous_real = None;
                previous_imaginary = None;
                previous_magnitude = None;
                continue;
            }

            let x = time_x(plot, sample.time_s, output.duration_s());
            let real = (x, center - sample.real * scale);
            let imaginary = (x, center - sample.imaginary * scale);
            let magnitude = (x, plot.bottom() - sample.magnitude * scale);

            if draw_complex {
                if let Some((px, py)) = previous_real {
                    list.line(px, py, real.0, real.1, 1.4, REAL);
                }
                if let Some((px, py)) = previous_imaginary {
                    list.line(px, py, imaginary.0, imaginary.1, 1.2, IMAGINARY);
                }
                previous_real = Some(real);
                previous_imaginary = Some(imaginary);
            }

            if let Some((px, py)) = previous_magnitude {
                list.line(px, py, magnitude.0, magnitude.1, 1.2, MAGNITUDE);
            }
            previous_magnitude = Some(magnitude);
        }

        let echo_x = time_x(plot, output.echo_time_s(), output.duration_s());
        list.vline(echo_x, plot.y, plot.bottom(), 1.0, ACCENT.with_alpha(0.60));
        fonts.draw_text(
            list,
            echo_x + 5.0,
            plot.y + 11.0,
            "TE",
            FontId::Mono,
            9.0,
            ACCENT,
        );

        let time_s = self.elapsed_time_s();
        list.vline(
            time_x(plot, time_s, output.duration_s()),
            plot.y,
            plot.bottom(),
            1.0,
            TEXT.with_alpha(0.55),
        );

        let mut legend_x = plot.x;
        if self.signal_view != SIGNAL_MAGNITUDE {
            fonts.draw_text(
                list,
                legend_x,
                plot.bottom() + 17.0,
                "real",
                FontId::Ui,
                9.0,
                REAL,
            );
            legend_x += 42.0;
            fonts.draw_text(
                list,
                legend_x,
                plot.bottom() + 17.0,
                "imag",
                FontId::Ui,
                9.0,
                IMAGINARY,
            );
            legend_x += 40.0;
        }
        fonts.draw_text(
            list,
            legend_x,
            plot.bottom() + 17.0,
            "magnitude",
            FontId::Ui,
            9.0,
            MAGNITUDE,
        );

        let range = format!("0 .. {:.1} ms", output.duration_s() * 1_000.0);
        let width = fonts.measure(&range, FontId::Mono, 9.0);
        fonts.draw_text(
            list,
            plot.right() - width,
            plot.bottom() + 17.0,
            &range,
            FontId::Mono,
            9.0,
            TEXT_DIM,
        );

        let current = signal_sample_at(samples, time_s);
        let value = format!(
            "S  {:+.3}  {:+.3}i  |S| {:.3}",
            current.real,
            current.imaginary,
            current.magnitude
        );
        let width = fonts.measure(&value, FontId::Mono, 9.0);
        fonts.draw_text(
            list,
            plot.right() - width,
            plot.y + 11.0,
            &value,
            FontId::Mono,
            9.0,
            TEXT_DIM,
        );

        list.pop_clip();
    }
}

impl Default for ViewState {
    fn default() -> Self {
        ViewState::new(&AppSettings::default())
    }
}

fn signed_degrees(radians: f32) -> f32 {
    let degrees = radians.to_degrees().rem_euclid(360.0);
    if degrees > 180.0 {
        degrees - 360.0
    } else {
        degrees
    }
}

fn signal_sample_at(samples: &[SignalSample], time_s: f32) -> SignalSample {
    if samples.is_empty() {
        return SignalSample::default();
    }

    match samples.binary_search_by(|sample| sample.time_s.total_cmp(&time_s)) {
        Ok(index) => samples[index],
        Err(0) => samples[0],
        Err(index) if index >= samples.len() => samples[samples.len() - 1],
        Err(index) => {
            let before = samples[index - 1];
            let after = samples[index];
            if time_s - before.time_s <= after.time_s - time_s {
                before
            } else {
                after
            }
        }
    }
}

fn time_x(plot: Rect, time_s: f32, duration_s: f32) -> f32 {
    plot.x + plot.w * (time_s / duration_s.max(1.0e-9)).clamp(0.0, 1.0)
}

fn draw_event(
    list: &mut DrawList,
    plot: Rect,
    baseline_y: f32,
    event: &FieldEvent,
    amplitude: f32,
    color: Color,
    duration_s: f32,
) {
    let baseline_y = plot.y + baseline_y;
    let x0 = time_x(plot, event.start_s, duration_s);
    let x1 = time_x(plot, event.end_s, duration_s).max(x0 + 1.0);
    let height = amplitude.abs().clamp(0.0, 1.0) * 15.0;

    if amplitude >= 0.0 {
        list.fill_rect(Rect::from_min_max(x0, baseline_y - height, x1, baseline_y), color);
    } else {
        list.fill_rect(Rect::from_min_max(x0, baseline_y, x1, baseline_y + height), color);
    }
}

