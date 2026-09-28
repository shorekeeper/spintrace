//! Spintrace configuration file.
//!
//! The file is read before the window, Vulkan instance and font atlas are
//! created. Construction settings can therefore affect those resources instead
//! of being applied after an unnecessary first allocation.
//!
//! Values intended for manual editing use display units: milliseconds, hertz,
//! degrees, millimetres and millitesla per metre. Conversion to SI units occurs
//! once while loading and the reverse conversion occurs while saving.
//!
//! Custom sequence events are stored in event.N sections. Those sections are
//! regenerated on save because their order and count belong to one list.
//! Unknown keys in ordinary sections and unknown sections outside the event
//! namespace remain untouched.

use std::path::{Path, PathBuf};

use crate::config::ini::Ini;
use crate::config::settings::{
    AnimCurve, AppearanceSettings, PresentModeCfg, RenderSettings, TabStyle,
};
use crate::config::ConfigEnum;
use crate::core::Result;
use crate::sim::{
    RfShape, SequenceEvent, SequenceEventKind, SequenceKind, SequenceProgram,
    SimulationConfig, MAX_SEQUENCE_EVENTS,
};

pub const DEFAULT_CONFIG_PATH: &str = "spintrace.ini";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalViewCfg {
    Complex,
    Magnitude,
    AcquiredOnly,
}

impl SignalViewCfg {
    pub const fn index(self) -> usize {
        match self {
            SignalViewCfg::Complex => 0,
            SignalViewCfg::Magnitude => 1,
            SignalViewCfg::AcquiredOnly => 2,
        }
    }

    pub const fn from_index(index: usize) -> SignalViewCfg {
        match index {
            1 => SignalViewCfg::Magnitude,
            2 => SignalViewCfg::AcquiredOnly,
            _ => SignalViewCfg::Complex,
        }
    }
}

impl ConfigEnum for SignalViewCfg {
    fn variants() -> &'static [&'static str] {
        &["complex", "magnitude", "acquired_only"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            SignalViewCfg::Complex => "complex",
            SignalViewCfg::Magnitude => "magnitude",
            SignalViewCfg::AcquiredOnly => "acquired_only",
        }
    }

    fn from_config(value: &str) -> Option<SignalViewCfg> {
        match value {
            "complex" => Some(SignalViewCfg::Complex),
            "magnitude" => Some(SignalViewCfg::Magnitude),
            "acquired_only" | "acquired-only" => Some(SignalViewCfg::AcquiredOnly),
            _ => None,
        }
    }
}

impl ConfigEnum for SequenceKind {
    fn variants() -> &'static [&'static str] {
        &["spin_echo", "gradient_echo", "custom"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            SequenceKind::SpinEcho => "spin_echo",
            SequenceKind::GradientEcho => "gradient_echo",
            SequenceKind::Custom => "custom",
        }
    }

    fn from_config(value: &str) -> Option<SequenceKind> {
        match value {
            "spin_echo" | "spin-echo" => Some(SequenceKind::SpinEcho),
            "gradient_echo" | "gradient-echo" => Some(SequenceKind::GradientEcho),
            "custom" => Some(SequenceKind::Custom),
            _ => None,
        }
    }
}

impl ConfigEnum for SequenceEventKind {
    fn variants() -> &'static [&'static str] {
        &["rf", "gradient", "adc"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            SequenceEventKind::Rf => "rf",
            SequenceEventKind::Gradient => "gradient",
            SequenceEventKind::Adc => "adc",
        }
    }

    fn from_config(value: &str) -> Option<SequenceEventKind> {
        match value {
            "rf" => Some(SequenceEventKind::Rf),
            "gradient" => Some(SequenceEventKind::Gradient),
            "adc" => Some(SequenceEventKind::Adc),
            _ => None,
        }
    }
}

impl ConfigEnum for RfShape {
    fn variants() -> &'static [&'static str] {
        &["rectangular", "gaussian", "sinc"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            RfShape::Rectangular => "rectangular",
            RfShape::Gaussian => "gaussian",
            RfShape::Sinc => "sinc",
        }
    }

    fn from_config(value: &str) -> Option<RfShape> {
        match value {
            "rectangular" | "rect" => Some(RfShape::Rectangular),
            "gaussian" => Some(RfShape::Gaussian),
            "sinc" => Some(RfShape::Sinc),
            _ => None,
        }
    }
}

/// Window state restored before the first frame.
#[derive(Debug, Clone)]
pub struct WindowSettings {
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
}

/// Font discovery, rasterization and atlas construction settings.
#[derive(Debug, Clone)]
pub struct FontSettings {
    pub ui_path: String,
    pub mono_path: String,
    pub atlas_size: u32,
    pub gamma: f32,
}

/// Complete persistent application state.
///
/// Resource settings are consumed during startup. Model, sequence and display
/// settings initialize the mutable state edited by the interface.
#[derive(Debug, Clone)]
pub struct AppSettings {
    pub window: WindowSettings,
    pub render: RenderSettings,
    pub font: FontSettings,
    pub appearance: AppearanceSettings,
    pub accent_rgb: u32,

    pub model: SimulationConfig,
    pub program: SequenceProgram,

    pub running: bool,
    pub time_scale: f32,
    pub vector_scale: f32,
    pub signal_view: SignalViewCfg,
    pub show_phase: bool,
    pub show_diagnostics: bool,
    pub frame_limit: u32,
    pub ui_scale: f32,
    pub ui_font_px: f32,
}

impl Default for AppSettings {
    fn default() -> AppSettings {
        let mut model = SimulationConfig::default();
        model.spin_count = 65_536;
        let program = SequenceProgram::preset(&model);

        AppSettings {
            window: WindowSettings {
                width: 1280,
                height: 800,
                maximized: false,
            },
            render: RenderSettings::default(),
            font: FontSettings {
                ui_path: String::new(),
                mono_path: String::new(),
                atlas_size: 1024,
                gamma: 1.25,
            },
            appearance: AppearanceSettings::default(),
            accent_rgb: 0x4A9EFF,

            model,
            program,

            running: true,
            time_scale: 0.15,
            vector_scale: 0.82,
            signal_view: SignalViewCfg::Complex,
            show_phase: true,
            show_diagnostics: true,
            frame_limit: 0,
            ui_scale: 1.0,
            ui_font_px: 12.0,
        }
    }
}

/// Retained INI document and its typed Spintrace representation.
///
/// Saving updates known keys in place, replaces indexed event sections and
/// preserves unrelated entries from the loaded file.
pub struct ConfigFile {
    path: PathBuf,
    document: Ini,
    pub settings: AppSettings,
}

impl ConfigFile {
    /// Loads an existing document or creates a documented default one.
    pub fn load(path: impl AsRef<Path>) -> Result<ConfigFile> {
        let path = path.as_ref().to_path_buf();
        let exists = path.is_file();
        let mut document = if exists {
            Ini::load(&path)?
        } else {
            default_document()
        };
        let settings = AppSettings::from_ini(&document);

        if exists {
            crate::log_info!("config", "loaded {}", path.display());
        } else {
            settings.write_to(&mut document);
            document.save(&path)?;
            crate::log_info!("config", "created {}", path.display());
        }

        Ok(ConfigFile { path, document, settings })
    }

    /// Writes known values into the retained document and saves it.
    pub fn save(&mut self) -> Result<()> {
        self.settings.write_to(&mut self.document);
        self.document.save(&self.path)?;
        crate::log_info!("config", "saved {}", self.path.display());
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl AppSettings {
    fn from_ini(ini: &Ini) -> AppSettings {
        let defaults = AppSettings::default();
        let mut out = defaults.clone();

        out.window.width =
            ini.get_u32_clamped("window", "width", defaults.window.width, 640, 7680);
        out.window.height =
            ini.get_u32_clamped("window", "height", defaults.window.height, 480, 4320);
        out.window.maximized =
            ini.get_bool("window", "maximized", defaults.window.maximized);

        out.render.validation =
            ini.get_bool("render", "validation", defaults.render.validation);
        out.render.device_index =
            ini.get_i32("render", "device_index", defaults.render.device_index);
        out.render.device_name =
            ini.get_string("render", "device_name", &defaults.render.device_name);
        out.render.log_device_info =
            ini.get_bool("render", "log_device_info", defaults.render.log_device_info);
        out.render.frames_in_flight = ini.get_u32_clamped(
            "render",
            "frames_in_flight",
            defaults.render.frames_in_flight,
            1,
            4,
        );
        out.render.swapchain_images = ini.get_u32_clamped(
            "render",
            "swapchain_images",
            defaults.render.swapchain_images,
            0,
            8,
        );
        out.render.present_mode =
            ini.get_enum("render", "present_mode", defaults.render.present_mode);
        out.render.vsync =
            ini.get_bool("render", "vsync", defaults.render.vsync);
        out.render.max_textures = ini.get_u32_clamped(
            "render",
            "max_textures",
            defaults.render.max_textures,
            4,
            4096,
        );
        out.render.vertex_buffer_kb = ini.get_u32_clamped(
            "render",
            "vertex_buffer_kb",
            defaults.render.vertex_buffer_kb,
            64,
            65_536,
        );
        out.render.index_buffer_kb = ini.get_u32_clamped(
            "render",
            "index_buffer_kb",
            defaults.render.index_buffer_kb,
            64,
            65_536,
        );
        out.render.background_rgb = read_rgb(
            ini,
            "render",
            "background",
            defaults.render.background_rgb,
        );

        out.font.ui_path =
            ini.get_string("font", "ui_path", &defaults.font.ui_path);
        out.font.mono_path =
            ini.get_string("font", "mono_path", &defaults.font.mono_path);
        out.font.atlas_size = ini.get_u32_clamped(
            "font",
            "atlas_size",
            defaults.font.atlas_size,
            256,
            4096,
        );
        out.font.gamma =
            ini.get_f32_clamped("font", "gamma", defaults.font.gamma, 0.5, 3.0);

        out.model.sequence =
            ini.get_enum("simulation", "sequence", defaults.model.sequence);
        out.model.spin_count = ini.get_usize(
            "simulation",
            "spins",
            defaults.model.spin_count,
        ).clamp(8, 1_048_576);
        out.model.observation_count = ini.get_usize(
            "simulation",
            "signal_samples",
            defaults.model.observation_count,
        ).clamp(32, 2_048);
        out.running =
            ini.get_bool("simulation", "running", defaults.running);
        out.time_scale = ini.get_f32_clamped(
            "simulation",
            "time_scale",
            defaults.time_scale,
            0.01,
            10.0,
        );

        out.model.t1_s = ini.get_f32_clamped(
            "ensemble",
            "t1_ms",
            defaults.model.t1_s * 1_000.0,
            0.001,
            1_000_000.0,
        ) * 0.001;
        out.model.t2_s = ini.get_f32_clamped(
            "ensemble",
            "t2_ms",
            defaults.model.t2_s * 1_000.0,
            0.001,
            1_000_000.0,
        ) * 0.001;
        out.model.t1_spread = ini.get_f32_clamped(
            "ensemble",
            "t1_spread_percent",
            defaults.model.t1_spread * 100.0,
            0.0,
            95.0,
        ) * 0.01;
        out.model.t2_spread = ini.get_f32_clamped(
            "ensemble",
            "t2_spread_percent",
            defaults.model.t2_spread * 100.0,
            0.0,
            95.0,
        ) * 0.01;
        out.model.center_offset_hz = ini.get_f32_clamped(
            "ensemble",
            "center_offset_hz",
            defaults.model.center_offset_hz,
            -100_000.0,
            100_000.0,
        );
        out.model.offset_span_hz = ini.get_f32_clamped(
            "ensemble",
            "offset_span_hz",
            defaults.model.offset_span_hz,
            0.0,
            200_000.0,
        );
        out.model.sample_extent_m[0] = ini.get_f32_clamped(
            "ensemble",
            "sample_width_mm",
            defaults.model.sample_extent_m[0] * 1_000.0,
            0.001,
            10_000.0,
        ) * 0.001;
        out.model.sample_extent_m[1] = ini.get_f32_clamped(
            "ensemble",
            "sample_height_mm",
            defaults.model.sample_extent_m[1] * 1_000.0,
            0.001,
            10_000.0,
        ) * 0.001;
        out.model.sample_extent_m[2] = ini.get_f32_clamped(
            "ensemble",
            "sample_depth_mm",
            defaults.model.sample_extent_m[2] * 1_000.0,
            0.001,
            10_000.0,
        ) * 0.001;

        out.model.te_s = ini.get_f32_clamped(
            "sequence",
            "te_ms",
            defaults.model.te_s * 1_000.0,
            0.001,
            10_000.0,
        ) * 0.001;
        out.model.excitation_flip_rad = ini.get_f32_clamped(
            "sequence",
            "excitation_flip_deg",
            defaults.model.excitation_flip_rad.to_degrees(),
            0.0,
            360.0,
        ).to_radians();
        out.model.refocus_flip_rad = ini.get_f32_clamped(
            "sequence",
            "refocus_flip_deg",
            defaults.model.refocus_flip_rad.to_degrees(),
            0.0,
            360.0,
        ).to_radians();
        out.model.rf_phase_rad = ini.get_f32_clamped(
            "sequence",
            "rf_phase_deg",
            signed_degrees(defaults.model.rf_phase_rad),
            -180.0,
            180.0,
        ).to_radians();
        out.model.rf_duration_s = ini.get_f32_clamped(
            "sequence",
            "rf_duration_ms",
            defaults.model.rf_duration_s * 1_000.0,
            0.001,
            1_000.0,
        ) * 0.001;
        out.model.gradient_amplitude_t_m = ini.get_f32_clamped(
            "sequence",
            "gradient_mt_m",
            defaults.model.gradient_amplitude_t_m * 1_000.0,
            0.0,
            100.0,
        ) * 0.001;
        out.model.adc_duration_s = ini.get_f32_clamped(
            "sequence",
            "adc_duration_ms",
            defaults.model.adc_duration_s * 1_000.0,
            0.001,
            10_000.0,
        ) * 0.001;
        out.model = out.model.sanitized();

        out.vector_scale = ini.get_f32_clamped(
            "display",
            "vector_scale",
            defaults.vector_scale,
            0.05,
            4.0,
        );
        out.signal_view =
            ini.get_enum("display", "signal_view", defaults.signal_view);
        out.show_phase =
            ini.get_bool("display", "phase_histogram", defaults.show_phase);
        out.show_diagnostics =
            ini.get_bool("display", "diagnostics", defaults.show_diagnostics);
        out.frame_limit = ini.get_u32_clamped(
            "display",
            "frame_limit",
            defaults.frame_limit,
            0,
            1_000,
        );
        out.ui_scale = ini.get_f32_clamped(
            "display",
            "ui_scale",
            defaults.ui_scale,
            0.5,
            4.0,
        );
        out.ui_font_px = ini.get_f32_clamped(
            "display",
            "font_size_px",
            defaults.ui_font_px,
            6.0,
            48.0,
        );

        out.accent_rgb =
            read_rgb(ini, "appearance", "accent", defaults.accent_rgb);
        out.appearance.data_background_rgb = read_rgb(
            ini,
            "appearance",
            "data_background",
            defaults.appearance.data_background_rgb,
        );
        out.appearance.grid_minor_alpha = ini.get_f32_clamped(
            "appearance",
            "grid_minor_alpha",
            defaults.appearance.grid_minor_alpha,
            0.0,
            1.0,
        );
        out.appearance.separator_alpha = ini.get_f32_clamped(
            "appearance",
            "separator_alpha",
            defaults.appearance.separator_alpha,
            0.0,
            1.0,
        );
        out.appearance.row_height = ini.get_f32_clamped(
            "appearance",
            "row_height",
            defaults.appearance.row_height,
            14.0,
            64.0,
        );
        out.appearance.gap = ini.get_f32_clamped(
            "appearance",
            "gap",
            defaults.appearance.gap,
            0.0,
            32.0,
        );
        out.appearance.panel_margin = ini.get_f32_clamped(
            "appearance",
            "panel_margin",
            defaults.appearance.panel_margin,
            0.0,
            64.0,
        );
        out.appearance.group_padding = ini.get_f32_clamped(
            "appearance",
            "group_padding",
            defaults.appearance.group_padding,
            0.0,
            64.0,
        );
        out.appearance.caption_height = ini.get_f32_clamped(
            "appearance",
            "caption_height",
            defaults.appearance.caption_height,
            18.0,
            80.0,
        );
        out.appearance.hint_scale = ini.get_f32_clamped(
            "appearance",
            "hint_scale",
            defaults.appearance.hint_scale,
            0.5,
            1.5,
        );
        out.appearance.focus_ring =
            ini.get_bool("appearance", "focus_ring", defaults.appearance.focus_ring);
        out.appearance.accent_hover =
            ini.get_bool("appearance", "accent_hover", defaults.appearance.accent_hover);
        out.appearance.group_tick =
            ini.get_bool("appearance", "group_tick", defaults.appearance.group_tick);
        out.appearance.tab_style =
            ini.get_enum("appearance", "tab_style", defaults.appearance.tab_style);
        out.appearance.value_column =
            ini.get_bool("appearance", "value_column", defaults.appearance.value_column);
        out.appearance.numeric_entry =
            ini.get_bool("appearance", "numeric_entry", defaults.appearance.numeric_entry);
        out.appearance.popup_shade = ini.get_f32_clamped(
            "appearance",
            "popup_shade",
            defaults.appearance.popup_shade,
            0.0,
            1.0,
        );
        out.appearance.group_activity = ini.get_bool(
            "appearance",
            "group_activity",
            defaults.appearance.group_activity,
        );
        out.appearance.keyboard_focus = ini.get_bool(
            "appearance",
            "keyboard_focus",
            defaults.appearance.keyboard_focus,
        );
        out.appearance.splitter_grip = ini.get_bool(
            "appearance",
            "splitter_grip",
            defaults.appearance.splitter_grip,
        );
        out.appearance.animate =
            ini.get_bool("appearance", "animate", defaults.appearance.animate);
        out.appearance.anim_ms = ini.get_f32_clamped(
            "appearance",
            "animation_ms",
            defaults.appearance.anim_ms,
            1.0,
            5_000.0,
        );
        out.appearance.anim_curve =
            ini.get_enum("appearance", "animation_curve", defaults.appearance.anim_curve);

        out.program = load_program(ini, &out.model, &defaults.program);
        out
    }

    fn write_to(&self, ini: &mut Ini) {
        ini.set_u32("window", "width", self.window.width);
        ini.set_u32("window", "height", self.window.height);
        ini.set_bool("window", "maximized", self.window.maximized);

        ini.set_bool("render", "validation", self.render.validation);
        ini.set_i32("render", "device_index", self.render.device_index);
        ini.set_string("render", "device_name", &self.render.device_name);
        ini.set_bool("render", "log_device_info", self.render.log_device_info);
        ini.set_u32("render", "frames_in_flight", self.render.frames_in_flight);
        ini.set_u32("render", "swapchain_images", self.render.swapchain_images);
        ini.set_enum("render", "present_mode", self.render.present_mode);
        ini.set_bool("render", "vsync", self.render.vsync);
        ini.set_u32("render", "max_textures", self.render.max_textures);
        ini.set_u32("render", "vertex_buffer_kb", self.render.vertex_buffer_kb);
        ini.set_u32("render", "index_buffer_kb", self.render.index_buffer_kb);
        write_rgb(ini, "render", "background", self.render.background_rgb);

        ini.set_string("font", "ui_path", &self.font.ui_path);
        ini.set_string("font", "mono_path", &self.font.mono_path);
        ini.set_u32("font", "atlas_size", self.font.atlas_size);
        ini.set_f32("font", "gamma", self.font.gamma);

        ini.set_enum("simulation", "sequence", self.model.sequence);
        ini.set_usize("simulation", "spins", self.model.spin_count);
        ini.set_usize(
            "simulation",
            "signal_samples",
            self.model.observation_count,
        );
        ini.set_bool("simulation", "running", self.running);
        ini.set_f32("simulation", "time_scale", self.time_scale);

        ini.set_f32("ensemble", "t1_ms", self.model.t1_s * 1_000.0);
        ini.set_f32("ensemble", "t2_ms", self.model.t2_s * 1_000.0);
        ini.set_f32(
            "ensemble",
            "t1_spread_percent",
            self.model.t1_spread * 100.0,
        );
        ini.set_f32(
            "ensemble",
            "t2_spread_percent",
            self.model.t2_spread * 100.0,
        );
        ini.set_f32(
            "ensemble",
            "center_offset_hz",
            self.model.center_offset_hz,
        );
        ini.set_f32(
            "ensemble",
            "offset_span_hz",
            self.model.offset_span_hz,
        );
        ini.set_f32(
            "ensemble",
            "sample_width_mm",
            self.model.sample_extent_m[0] * 1_000.0,
        );
        ini.set_f32(
            "ensemble",
            "sample_height_mm",
            self.model.sample_extent_m[1] * 1_000.0,
        );
        ini.set_f32(
            "ensemble",
            "sample_depth_mm",
            self.model.sample_extent_m[2] * 1_000.0,
        );

        ini.set_f32("sequence", "te_ms", self.model.te_s * 1_000.0);
        ini.set_f32(
            "sequence",
            "excitation_flip_deg",
            self.model.excitation_flip_rad.to_degrees(),
        );
        ini.set_f32(
            "sequence",
            "refocus_flip_deg",
            self.model.refocus_flip_rad.to_degrees(),
        );
        ini.set_f32(
            "sequence",
            "rf_phase_deg",
            signed_degrees(self.model.rf_phase_rad),
        );
        ini.set_f32(
            "sequence",
            "rf_duration_ms",
            self.model.rf_duration_s * 1_000.0,
        );
        ini.set_f32(
            "sequence",
            "gradient_mt_m",
            self.model.gradient_amplitude_t_m * 1_000.0,
        );
        ini.set_f32(
            "sequence",
            "adc_duration_ms",
            self.model.adc_duration_s * 1_000.0,
        );

        let mut program = self.program.clone();
        program.sanitize();
        ini.set_f32("sequence", "duration_ms", program.duration_s * 1_000.0);
        ini.set_f32(
            "sequence",
            "echo_marker_ms",
            program.echo_time_s * 1_000.0,
        );
        ini.set_u32("sequence", "event_count", program.events.len() as u32);

        ini.set_enum("display", "signal_view", self.signal_view);
        ini.set_f32("display", "vector_scale", self.vector_scale);
        ini.set_bool("display", "phase_histogram", self.show_phase);
        ini.set_bool("display", "diagnostics", self.show_diagnostics);
        ini.set_u32("display", "frame_limit", self.frame_limit);
        ini.set_f32("display", "ui_scale", self.ui_scale);
        ini.set_f32("display", "font_size_px", self.ui_font_px);

        write_rgb(ini, "appearance", "accent", self.accent_rgb);
        write_rgb(
            ini,
            "appearance",
            "data_background",
            self.appearance.data_background_rgb,
        );
        ini.set_f32(
            "appearance",
            "grid_minor_alpha",
            self.appearance.grid_minor_alpha,
        );
        ini.set_f32(
            "appearance",
            "separator_alpha",
            self.appearance.separator_alpha,
        );
        ini.set_f32("appearance", "row_height", self.appearance.row_height);
        ini.set_f32("appearance", "gap", self.appearance.gap);
        ini.set_f32(
            "appearance",
            "panel_margin",
            self.appearance.panel_margin,
        );
        ini.set_f32(
            "appearance",
            "group_padding",
            self.appearance.group_padding,
        );
        ini.set_f32(
            "appearance",
            "caption_height",
            self.appearance.caption_height,
        );
        ini.set_f32("appearance", "hint_scale", self.appearance.hint_scale);
        ini.set_bool("appearance", "focus_ring", self.appearance.focus_ring);
        ini.set_bool(
            "appearance",
            "accent_hover",
            self.appearance.accent_hover,
        );
        ini.set_bool("appearance", "group_tick", self.appearance.group_tick);
        ini.set_enum("appearance", "tab_style", self.appearance.tab_style);
        ini.set_bool(
            "appearance",
            "value_column",
            self.appearance.value_column,
        );
        ini.set_bool(
            "appearance",
            "numeric_entry",
            self.appearance.numeric_entry,
        );
        ini.set_f32(
            "appearance",
            "popup_shade",
            self.appearance.popup_shade,
        );
        ini.set_bool(
            "appearance",
            "group_activity",
            self.appearance.group_activity,
        );
        ini.set_bool(
            "appearance",
            "keyboard_focus",
            self.appearance.keyboard_focus,
        );
        ini.set_bool(
            "appearance",
            "splitter_grip",
            self.appearance.splitter_grip,
        );
        ini.set_bool("appearance", "animate", self.appearance.animate);
        ini.set_f32("appearance", "animation_ms", self.appearance.anim_ms);
        ini.set_enum(
            "appearance",
            "animation_curve",
            self.appearance.anim_curve,
        );

        ini.remove_sections_with_prefix("event.");
        for (index, event) in program.events.iter().enumerate() {
            write_event(ini, index + 1, event);
        }
    }
}

fn load_program(
    ini: &Ini,
    model: &SimulationConfig,
    default: &SequenceProgram,
) -> SequenceProgram {
    if ini.raw("sequence", "event_count").is_none() {
        return SequenceProgram::preset(model);
    }

    let duration_s = ini.get_f32_clamped(
        "sequence",
        "duration_ms",
        default.duration_s * 1_000.0,
        1.0,
        10_000.0,
    ) * 0.001;
    let echo_time_s = ini.get_f32_clamped(
        "sequence",
        "echo_marker_ms",
        default.echo_time_s * 1_000.0,
        0.0,
        duration_s * 1_000.0,
    ) * 0.001;
    let count = ini.get_u32_clamped(
        "sequence",
        "event_count",
        default.events.len() as u32,
        0,
        MAX_SEQUENCE_EVENTS as u32,
    ) as usize;

    let mut program = SequenceProgram::new(duration_s, echo_time_s);
    for index in 0..count {
        let section = format!("event.{}", index + 1);
        if !ini.has_section(&section) {
            crate::log_warn!("config", "missing [{}]", section);
            continue;
        }

        let kind = ini.get_enum(&section, "type", SequenceEventKind::Rf);
        let mut event = SequenceEvent {
            kind,
            start_s: ini.get_f32_clamped(
                &section,
                "start_ms",
                0.0,
                0.0,
                duration_s * 1_000.0,
            ) * 0.001,
            duration_s: ini.get_f32_clamped(
                &section,
                "duration_ms",
                0.5,
                0.001,
                duration_s * 1_000.0,
            ) * 0.001,
            rf_flip_rad: ini.get_f32_clamped(
                &section,
                "flip_deg",
                90.0,
                0.0,
                360.0,
            ).to_radians(),
            rf_phase_rad: ini.get_f32_clamped(
                &section,
                "phase_deg",
                0.0,
                -180.0,
                180.0,
            ).to_radians(),
            rf_shape: ini.get_enum(
                &section,
                "shape",
                RfShape::Rectangular,
            ),
            rf_samples: ini.get_u32_clamped(
                &section,
                "samples",
                32,
                1,
                256,
            ),
            gradient_t_m: [
                ini.get_f32_clamped(
                    &section,
                    "gx_mt_m",
                    0.0,
                    -100.0,
                    100.0,
                ) * 0.001,
                ini.get_f32_clamped(
                    &section,
                    "gy_mt_m",
                    0.0,
                    -100.0,
                    100.0,
                ) * 0.001,
                ini.get_f32_clamped(
                    &section,
                    "gz_mt_m",
                    0.0,
                    -100.0,
                    100.0,
                ) * 0.001,
            ],
        };
        event.sanitize(duration_s);
        program.events.push(event);
    }

    program.sanitize();
    program
}

fn write_event(ini: &mut Ini, number: usize, event: &SequenceEvent) {
    let section = format!("event.{}", number);
    ini.set_enum(&section, "type", event.kind);
    ini.set_f32(&section, "start_ms", event.start_s * 1_000.0);
    ini.set_f32(&section, "duration_ms", event.duration_s * 1_000.0);
    ini.set_f32(&section, "flip_deg", event.rf_flip_rad.to_degrees());
    ini.set_f32(
        &section,
        "phase_deg",
        signed_degrees(event.rf_phase_rad),
    );
    ini.set_enum(&section, "shape", event.rf_shape);
    ini.set_u32(&section, "samples", event.rf_samples);
    ini.set_f32(
        &section,
        "gx_mt_m",
        event.gradient_t_m[0] * 1_000.0,
    );
    ini.set_f32(
        &section,
        "gy_mt_m",
        event.gradient_t_m[1] * 1_000.0,
    );
    ini.set_f32(
        &section,
        "gz_mt_m",
        event.gradient_t_m[2] * 1_000.0,
    );
}

fn default_document() -> Ini {
    let mut ini = Ini::new();

    ini.comment("", "Spintrace configuration.");
    ini.comment("", "Changes are loaded at startup and saved on clean exit.");

    ini.comment("window", "Initial client size and restored window state.");

    ini.comment("render", "Vulkan construction and presentation settings.");
    ini.comment(
        "render",
        "Auto uses vsync to choose FIFO or the best available non-blocking mode.",
    );
    ini.comment("render", "A device index overrides a device name match.");

    ini.comment("font", "Empty paths select installed system fonts.");
    ini.comment("font", "The atlas size is fixed for the process lifetime.");

    ini.comment("simulation", "Global model and playback settings.");
    ini.comment(
        "simulation",
        "Sequence is spin_echo, gradient_echo or custom.",
    );

    ini.comment("ensemble", "Relaxation, frequency and sample geometry.");
    ini.comment("ensemble", "Spread values are full symmetric percentages.");

    ini.comment("sequence", "Preset parameters and editable timeline metadata.");
    ini.comment(
        "sequence",
        "Times use milliseconds, angles use degrees and gradients use mT/m.",
    );

    ini.comment("display", "Signal, interface and diagnostic presentation.");

    ini.comment("appearance", "Interface palette, metrics and interaction style.");
    ini.comment("appearance", "Colours use #RRGGBB.");

    ini
}

fn read_rgb(ini: &Ini, section: &str, key: &str, default: u32) -> u32 {
    let value = match ini.raw(section, key) {
        Some(value) => value.trim(),
        None => return default,
    };
    let digits = value
        .strip_prefix('#')
        .or_else(|| value.strip_prefix("0x"))
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);

    match u32::from_str_radix(digits, 16) {
        Ok(rgb) if rgb <= 0xFF_FFFF => rgb,
        _ => {
            crate::log_warn!(
                "config",
                "[{}] {}: bad colour '{}'",
                section,
                key,
                value
            );
            default
        }
    }
}

fn write_rgb(ini: &mut Ini, section: &str, key: &str, value: u32) {
    ini.set_string(section, key, &format!("#{:06X}", value & 0xFF_FFFF));
}

fn signed_degrees(radians: f32) -> f32 {
    let degrees = radians.to_degrees().rem_euclid(360.0);
    if degrees > 180.0 {
        degrees - 360.0
    } else {
        degrees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_and_events_round_trip() {
        let mut settings = AppSettings::default();
        settings.render.present_mode = PresentModeCfg::Mailbox;
        settings.render.vsync = false;
        settings.model.sequence = SequenceKind::Custom;
        settings.signal_view = SignalViewCfg::AcquiredOnly;
        settings.program.events.clear();
        settings.program.events.push(SequenceEvent::rf(
            0.004,
            0.001,
            std::f32::consts::PI * 0.75,
            -0.4,
            RfShape::Sinc,
            48,
        ));
        settings.program.events.push(SequenceEvent::adc(0.020, 0.006));

        let mut ini = default_document();
        settings.write_to(&mut ini);
        let loaded = AppSettings::from_ini(&Ini::parse(&ini.to_text()));

        assert_eq!(loaded.render.present_mode, PresentModeCfg::Mailbox);
        assert!(!loaded.render.vsync);
        assert_eq!(loaded.model.sequence, SequenceKind::Custom);
        assert_eq!(loaded.signal_view, SignalViewCfg::AcquiredOnly);
        assert_eq!(loaded.program.events.len(), 2);
        assert_eq!(loaded.program.events[0].rf_shape, RfShape::Sinc);
        assert_eq!(loaded.program.events[0].rf_samples, 48);
    }

    #[test]
    fn writing_settings_preserves_foreign_entries() {
        let mut ini = Ini::parse(
            "[foreign]\nopaque = keep\n\n[event.99]\ntype = adc\n",
        );
        AppSettings::default().write_to(&mut ini);

        assert_eq!(ini.raw("foreign", "opaque"), Some("keep"));
        assert!(!ini.has_section("event.99"));
    }
}