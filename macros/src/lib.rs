//! Procedural macros used by the ECS runtime.
//!
//! This crate keeps code generation small and predictable: derives generate
//! trait implementations expected by the runtime crate, while `all_tuples!`
//! expands repetitive tuple implementations used across the ECS internals.
//!
//! # Exported derives
//!
//! - `#[derive(Component)]`: marks a type as an ECS component.
//! - `#[derive(Bundle)]`: marks a struct as a spawnable component bundle.
//!
//! # Exported function-like macros
//!
//! - `all_tuples!`: expands an implementation macro over tuple arities.

// The MSVC linker prints informational "Creando biblioteca" lines to stdout on
// every proc-macro build; they carry no diagnostics and only add noise.
#![allow(linker_messages)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::{format_ident, quote};

mod bundle;
mod component;
mod variadic;

/// Package name of the runtime crate that generated code refers to.
const CORE_CRATE: &str = "millarecs";

/// Resolves the path that generated code should use to refer to the runtime crate.
///
/// When the macro is expanded inside the runtime crate itself, `crate` is used.
/// When it is expanded from a dependent crate, the discovered dependency name is
/// converted to an absolute path. If crate discovery fails, the default package
/// name is used as a fallback so diagnostics still point at the generated code.
fn get_ecs_path() -> TokenStream2 {
    match crate_name(CORE_CRATE) {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{}", name);
            quote!(::#ident)
        }
        Err(_) => {
            let ident = format_ident!("{}", CORE_CRATE);
            quote!(::#ident)
        }
    }
}

/// Derives the ECS `Component` marker trait.
///
/// The derive does not add `Clone` or `Default`; components are moved into
/// storage and those capabilities remain explicit choices of the component
/// type. Structs, enums, and unions are accepted when they satisfy the runtime
/// `Send + Sync + 'static` requirements.
///
/// ```rust,ignore
/// #[derive(Component)]
/// struct Position {
///     x: f32,
///     y: f32,
/// }
///
/// #[derive(Component)]
/// struct Marker;
/// ```
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    component::derive_component(input)
}

/// Derives the ECS `Bundle` trait for named structs.
///
/// Each field becomes one erased bundle item. The generated implementation
/// delegates duplicate checking, registration lookup, and ownership handling to
/// the runtime crate, keeping the macro responsible only for structural code.
///
/// Tuple structs, unit structs, enums, and unions are rejected because the
/// generated implementation must destructure fields by name.
///
/// ```rust,ignore
/// #[derive(Bundle)]
/// struct PlayerBundle {
///     position: Position,
///     velocity: Velocity,
/// }
/// ```
#[proc_macro_derive(Bundle)]
pub fn derive_bundle(input: TokenStream) -> TokenStream {
    bundle::derive_bundle(input)
}

/// Invokes another macro once for every tuple arity in an inclusive range.
///
/// The target macro receives an `#[allow(non_snake_case)]` attribute followed by
/// generated identifiers using the requested prefix. This is useful for tuple
/// trait implementations where the implementation body is otherwise identical.
/// Several prefixes interleave per arity position: with prefixes `S, M`, the
/// arity-two call receives `S0, M0, S1, M1`.
///
/// ```rust,ignore
/// all_tuples!(impl_bundle_tuple, 0, 3, T);
///
/// // Expands to calls equivalent to:
/// // impl_bundle_tuple!(#[allow(non_snake_case)]);
/// // impl_bundle_tuple!(#[allow(non_snake_case)] T0);
/// // impl_bundle_tuple!(#[allow(non_snake_case)] T0, T1);
/// // impl_bundle_tuple!(#[allow(non_snake_case)] T0, T1, T2);
/// ```
#[proc_macro]
pub fn all_tuples(input: TokenStream) -> TokenStream {
    variadic::all_tuples(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_ecs_path_mentions_runtime_crate() {
        let path = get_ecs_path().to_string();

        assert!(path == "crate" || path.contains(CORE_CRATE));
    }
}
