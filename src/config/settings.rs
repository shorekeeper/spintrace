//! Renderer and interface construction settings.
//!
//! RenderSettings controls resources that must be selected before Vulkan
//! objects are created. AppearanceSettings contains the palette, dimensions
//! and interaction switches consumed by the immediate mode GUI.
//!
//! Mutable simulation and sequence values are stored by AppSettings and copied
//! into the application view after construction.

use crate::config::ConfigEnum;

/// Requested swapchain presentation policy.
///
/// Auto selects from the available modes according to the VSync setting. Every
/// explicit variant requests that mode and falls back through swapchain
/// selection when the surface does not expose it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentModeCfg {
    Auto,
    Fifo,
    FifoRelaxed,
    Mailbox,
    Immediate,
}

impl PresentModeCfg {
    pub const LABELS: [&'static str; 5] = [
        "Auto",
        "FIFO",
        "FIFO relaxed",
        "Mailbox",
        "Immediate",
    ];

    pub const fn index(self) -> usize {
        match self {
            PresentModeCfg::Auto => 0,
            PresentModeCfg::Fifo => 1,
            PresentModeCfg::FifoRelaxed => 2,
            PresentModeCfg::Mailbox => 3,
            PresentModeCfg::Immediate => 4,
        }
    }

    pub const fn from_index(index: usize) -> PresentModeCfg {
        match index {
            1 => PresentModeCfg::Fifo,
            2 => PresentModeCfg::FifoRelaxed,
            3 => PresentModeCfg::Mailbox,
            4 => PresentModeCfg::Immediate,
            _ => PresentModeCfg::Auto,
        }
    }
}

impl ConfigEnum for PresentModeCfg {
    fn variants() -> &'static [&'static str] {
        &["auto", "fifo", "fifo_relaxed", "mailbox", "immediate"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            PresentModeCfg::Auto => "auto",
            PresentModeCfg::Fifo => "fifo",
            PresentModeCfg::FifoRelaxed => "fifo_relaxed",
            PresentModeCfg::Mailbox => "mailbox",
            PresentModeCfg::Immediate => "immediate",
        }
    }

    fn from_config(value: &str) -> Option<PresentModeCfg> {
        match value {
            "auto" => Some(PresentModeCfg::Auto),
            "fifo" => Some(PresentModeCfg::Fifo),
            "fifo_relaxed" | "fifo-relaxed" => Some(PresentModeCfg::FifoRelaxed),
            "mailbox" => Some(PresentModeCfg::Mailbox),
            "immediate" => Some(PresentModeCfg::Immediate),
            _ => None,
        }
    }
}

/// Selection marker used by tab strips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabStyle {
    Underline,
    Box,
}

/// Time mapping applied to finite GUI transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

impl AnimCurve {
    pub fn apply(self, value: f32) -> f32 {
        let value = value.clamp(0.0, 1.0);
        match self {
            AnimCurve::Linear => value,
            AnimCurve::EaseIn => value * value,
            AnimCurve::EaseOut => 1.0 - (1.0 - value) * (1.0 - value),
            AnimCurve::EaseInOut => {
                if value < 0.5 {
                    2.0 * value * value
                } else {
                    1.0 - (-2.0 * value + 2.0).powi(2) * 0.5
                }
            }
        }
    }
}

impl ConfigEnum for TabStyle {
    fn variants() -> &'static [&'static str] {
        &["underline", "box"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            TabStyle::Underline => "underline",
            TabStyle::Box => "box",
        }
    }

    fn from_config(value: &str) -> Option<TabStyle> {
        match value {
            "underline" => Some(TabStyle::Underline),
            "box" => Some(TabStyle::Box),
            _ => None,
        }
    }
}

impl ConfigEnum for AnimCurve {
    fn variants() -> &'static [&'static str] {
        &["linear", "ease_in", "ease_out", "ease_in_out"]
    }

    fn to_config(&self) -> &'static str {
        match self {
            AnimCurve::Linear => "linear",
            AnimCurve::EaseIn => "ease_in",
            AnimCurve::EaseOut => "ease_out",
            AnimCurve::EaseInOut => "ease_in_out",
        }
    }

    fn from_config(value: &str) -> Option<AnimCurve> {
        match value {
            "linear" => Some(AnimCurve::Linear),
            "ease_in" | "ease-in" => Some(AnimCurve::EaseIn),
            "ease_out" | "ease-out" => Some(AnimCurve::EaseOut),
            "ease_in_out" | "ease-in-out" => Some(AnimCurve::EaseInOut),
            _ => None,
        }
    }
}

/// Values copied into the GUI theme.
///
/// Metrics are logical units at ninety six dpi. Colours that are not exposed
/// here belong to the fixed spintrace palette and are selected by Theme.
#[derive(Debug, Clone, Copy)]
pub struct AppearanceSettings {
    pub data_background_rgb: u32,
    pub grid_minor_alpha: f32,
    pub separator_alpha: f32,
    pub row_height: f32,
    pub gap: f32,
    pub panel_margin: f32,
    pub group_padding: f32,
    pub caption_height: f32,
    pub hint_scale: f32,
    pub focus_ring: bool,
    pub accent_hover: bool,
    pub group_tick: bool,
    pub tab_style: TabStyle,
    pub value_column: bool,
    pub numeric_entry: bool,
    pub popup_shade: f32,
    pub group_activity: bool,
    pub keyboard_focus: bool,
    pub splitter_grip: bool,
    pub animate: bool,
    pub anim_ms: f32,
    pub anim_curve: AnimCurve,
}

impl Default for AppearanceSettings {
    fn default() -> AppearanceSettings {
        AppearanceSettings {
            data_background_rgb: 0x0A0A0C,
            grid_minor_alpha: 0.45,
            separator_alpha: 0.70,
            row_height: 22.0,
            gap: 4.0,
            panel_margin: 6.0,
            group_padding: 6.0,
            caption_height: 28.0,
            hint_scale: 0.85,
            focus_ring: true,
            accent_hover: false,
            group_tick: false,
            tab_style: TabStyle::Underline,
            value_column: true,
            numeric_entry: true,
            popup_shade: 0.18,
            group_activity: true,
            keyboard_focus: true,
            splitter_grip: true,
            animate: true,
            anim_ms: 120.0,
            anim_curve: AnimCurve::EaseOut,
        }
    }
}

/// Vulkan resource and presentation settings read before renderer creation.
#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub validation: bool,
    pub device_index: i32,
    pub device_name: String,
    pub log_device_info: bool,
    pub frames_in_flight: u32,
    pub swapchain_images: u32,
    pub present_mode: PresentModeCfg,
    pub vsync: bool,
    pub max_textures: u32,
    pub vertex_buffer_kb: u32,
    pub index_buffer_kb: u32,
    pub background_rgb: u32,
}

impl Default for RenderSettings {
    fn default() -> RenderSettings {
        RenderSettings {
            validation: cfg!(debug_assertions),
            device_index: -1,
            device_name: String::new(),
            log_device_info: true,
            frames_in_flight: 2,
            swapchain_images: 0,
            present_mode: PresentModeCfg::Auto,
            vsync: true,
            max_textures: 64,
            vertex_buffer_kb: 512,
            index_buffer_kb: 512,
            background_rgb: 0x111418,
        }
    }
}