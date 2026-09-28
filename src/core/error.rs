//! Application error type.
//!
//! Errors carry a stable category and may retain a native result code. The
//! message describes the failed operation rather than the layer that reports
//! it, so callers can add context without decoding an opaque integer.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Platform,
    Vulkan,
    Simulation,
    Data,
    Font,
    Config,
}

impl Category {
    fn name(self) -> &'static str {
        match self {
            Category::Platform => "platform",
            Category::Vulkan => "vulkan",
            Category::Simulation => "simulation",
            Category::Data => "data",
            Category::Font => "font",
            Category::Config => "config",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    pub category: Category,
    pub message: String,
    pub code: Option<i64>,
}

impl Error {
    pub fn new(category: Category, message: impl Into<String>) -> Error {
        Error { category, message: message.into(), code: None }
    }

    pub fn with_code(category: Category, message: impl Into<String>, code: i64) -> Error {
        Error { category, message: message.into(), code: Some(code) }
    }

    pub fn platform(message: impl Into<String>) -> Error {
        Error::new(Category::Platform, message)
    }

    pub fn vulkan(message: impl Into<String>) -> Error {
        Error::new(Category::Vulkan, message)
    }

    pub fn simulation(message: impl Into<String>) -> Error {
        Error::new(Category::Simulation, message)
    }

    pub fn data(message: impl Into<String>) -> Error {
        Error::new(Category::Data, message)
    }

    pub fn font(message: impl Into<String>) -> Error {
        Error::new(Category::Font, message)
    }

    pub fn config(message: impl Into<String>) -> Error {
        Error::new(Category::Config, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "{}: {} ({})", self.category.name(), self.message, code),
            None => write!(f, "{}: {}", self.category.name(), self.message),
        }
    }
}

impl std::error::Error for Error {}