//! Native window and input events.
//!
//! Platform messages are translated into a compact event vocabulary before
//! they reach the GUI. The rest of the application does not inspect virtual
//! key codes, packed coordinates or Win32 window messages.
//!
//! Modifiers are sampled when an event is produced. A pointer gesture therefore
//! keeps the state that accompanied the native message rather than reading keys
//! later during frame construction.

#[cfg(not(target_os = "windows"))]
compile_error!("spintrace currently requires Windows");

pub mod win32;

pub use win32::Window;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    X1,
    X2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Tab,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Enter,
    Escape,
    Space,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorKind {
    Arrow,
    Hand,
    Text,
    ResizeHorizontal,
    ResizeVertical,
}

#[derive(Debug, Clone, Copy)]
pub enum Event {
    MouseMove {
        x: f32,
        y: f32,
        mods: Modifiers,
    },
    MouseButton {
        button: MouseButton,
        pressed: bool,
        x: f32,
        y: f32,
        mods: Modifiers,
    },
    MouseDoubleClick {
        button: MouseButton,
        x: f32,
        y: f32,
    },
    MouseWheel {
        delta_y: f32,
        delta_x: f32,
        x: f32,
        y: f32,
        mods: Modifiers,
    },
    MouseLeave,
    Key {
        key: Key,
        pressed: bool,
        repeat: bool,
        mods: Modifiers,
    },
    Text(char),
    Focus(bool),
    Resized {
        width: u32,
        height: u32,
    },
}