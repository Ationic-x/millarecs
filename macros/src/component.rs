//! Implementation of the `Component` derive macro.
//!
//! The derive expands to the runtime marker trait and tightens generic
//! parameters with the thread-safety bounds required by the scheduler.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, parse_quote, DeriveInput};

use crate::get_ecs_path;

/// Expands `#[derive(Component)]` into the runtime marker implementation.
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match expand_component(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Builds the generated `Component` implementation.
///
/// Generic bounds are tightened rather than assumed: every type parameter
/// receives `'static + Send + Sync` and every lifetime parameter receives
/// `'static`, so the empty marker implementation only exists for
/// instantiations that systems can safely share across worker threads.
/// Pre-existing `where` clauses are preserved untouched.
fn expand_component(input: DeriveInput) -> syn::Result<TokenStream2> {
    let ecs_path = get_ecs_path();
    let ty = input.ident;
    let mut generics = input.generics;

    for parameter in generics.type_params_mut() {
        parameter.bounds.push(parse_quote!('static));
        parameter.bounds.push(parse_quote!(::core::marker::Send));
        parameter.bounds.push(parse_quote!(::core::marker::Sync));
    }

    for parameter in generics.lifetimes_mut() {
        parameter.bounds.push(parse_quote!('static));
    }

    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics
            #ecs_path::__private::Component
            for #ty #type_generics
            #where_clause
        {}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expanded(input: DeriveInput) -> String {
        expand_component(input).unwrap().to_string()
    }

    #[test]
    fn named_struct_only_implements_component() {
        let input: DeriveInput = parse_quote! {
            struct Position {
                x: f32,
                y: f32,
            }
        };

        let output = expanded(input);

        assert!(output.contains("impl :: millarecs :: __private :: Component for Position"));
        assert!(!output.contains("Clone"));
        assert!(!output.contains("Default"));
    }

    #[test]
    fn tuple_struct_is_supported() {
        let input: DeriveInput = parse_quote! {
            struct Velocity(f32, f32);
        };

        let output = expanded(input);

        assert!(output.contains("impl :: millarecs :: __private :: Component for Velocity"));
    }

    #[test]
    fn unit_struct_is_supported() {
        let input: DeriveInput = parse_quote! {
            struct Player;
        };

        let output = expanded(input);

        assert!(output.contains("impl :: millarecs :: __private :: Component for Player"));
    }

    #[test]
    fn generic_type_parameters_receive_thread_safety_bounds() {
        let input: DeriveInput = parse_quote! {
            struct Wrapper<T, U> {
                left: T,
                right: U,
            }
        };

        let output = expanded(input);

        assert!(
            output.contains("T : 'static + :: core :: marker :: Send + :: core :: marker :: Sync")
        );
        assert!(
            output.contains("U : 'static + :: core :: marker :: Send + :: core :: marker :: Sync")
        );
        assert!(output.contains("Wrapper < T , U >"));
    }

    #[test]
    fn existing_where_clause_is_preserved() {
        let input: DeriveInput = parse_quote! {
            struct Wrapped<T>
            where
                T: Send,
            {
                value: T,
            }
        };

        let output = expanded(input);

        assert!(output.contains("where T : Send"));
        assert!(output.contains(
            "impl < T : 'static + :: core :: marker :: Send + :: core :: marker :: Sync >"
        ));
    }

    #[test]
    fn enum_input_is_supported() {
        let input: DeriveInput = parse_quote! {
            enum State {
                A,
                B,
            }
        };

        let output = expanded(input);

        assert!(output.contains("impl :: millarecs :: __private :: Component for State"));
    }

    #[test]
    fn union_input_is_supported() {
        let input: DeriveInput = parse_quote! {
            union Value {
                a: u32,
                b: f32,
            }
        };

        let output = expanded(input);

        assert!(output.contains("impl :: millarecs :: __private :: Component for Value"));
    }
}
