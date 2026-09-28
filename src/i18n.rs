//! Display string catalogue.
//!
//! The widget layer addresses text through stable keys. Spintrace currently
//! uses the key itself as the English display string, but the catalogue remains
//! a separate type so localization can be added without changing widget
//! identity or call sites.

pub struct Catalog {
    language: String,
}

impl Catalog {
    pub fn passthrough(language: impl Into<String>) -> Catalog {
        Catalog { language: language.into() }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn get<'a>(&'a self, key: &'a str) -> &'a str {
        key
    }
}

pub fn current() -> Catalog {
    Catalog::passthrough("en")
}