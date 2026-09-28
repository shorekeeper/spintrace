//! Spintrace application entry point.
//!
//! Subsystems are kept as crate modules rather than hidden behind an external
//! framework. The executable owns the Win32 loop, Vulkan renderer, GUI, font
//! cache and simulation model directly.

mod app;
mod config;
mod core;
mod font;
mod gui;
mod i18n;
mod platform;
mod render;
mod sim;
mod view;

fn main() {
    if let Err(error) = app::run() {
        eprintln!("{}", error);
        std::process::exit(1);
    }
}