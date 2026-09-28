//! Shared error and logging facilities.

pub mod error;

pub use error::{Error, Result};

#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)*) => {{
        if cfg!(debug_assertions) {
            eprintln!("[debug] [{}] {}", $target, format_args!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)*) => {{
        eprintln!("[info] [{}] {}", $target, format_args!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)*) => {{
        eprintln!("[warn] [{}] {}", $target, format_args!($($arg)*));
    }};
}

#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)*) => {{
        eprintln!("[error] [{}] {}", $target, format_args!($($arg)*));
    }};
}