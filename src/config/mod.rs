//! Persistent and runtime configuration.
//!
//! Ini stores the document without discarding unknown entries. AppSettings maps
//! the known keys into typed values used by the window, renderer, interface and
//! simulation subsystems.
//!
//! ConfigEnum keeps file values independent of display labels. A translated or
//! reformatted control may therefore retain the same stored representation.

pub mod file;
pub mod ini;
pub mod settings;

pub use file::{
    AppSettings, ConfigFile, FontSettings, SignalViewCfg, WindowSettings,
    DEFAULT_CONFIG_PATH,
};

pub trait ConfigEnum: Copy + Sized {
    fn variants() -> &'static [&'static str];
    fn to_config(&self) -> &'static str;
    fn from_config(value: &str) -> Option<Self>;
}