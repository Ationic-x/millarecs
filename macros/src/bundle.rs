//! Implementation of the `Bundle` derive macro.
//!
//! The generated code turns named struct fields into erased bundle items and
//! delegates ownership, duplicate detection, and component registration checks
//! to the runtime crate.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Error, Fields, Ident, Type};

use crate::get_ecs_path;

/// Expands `#[derive(Bundle)]` for named structs.
pub fn derive_bundle(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match expand_bundle(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Builds the generated `Bundle` implementation.
///
/// The generated code destructures the bundle by value, wraps each field in
/// `ManuallyDrop`, converts every field into an erased bundle item, and finally
/// calls the runtime helper that owns duplicate checks and safe drop handling.
/// Enums, unions, and structs without named fields are rejected because both
/// generated methods address components through their field names.
fn expand_bundle(input: DeriveInput) -> syn::Result<TokenStream2> {
    let ecs_path = get_ecs_path();
    let struct_name = input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(Error::new_spanned(
                struct_name,
                "Bundle can only derive from structs",
            ))
        }
    };

    let fields = match data.fields {
        Fields::Named(fields) => fields.named,
        _ => {
            return Err(Error::new_spanned(
                struct_name,
                "Bundle require named fields",
            ))
        }
    };

    let field_idents: Vec<Ident> = fields
        .iter()
        .map(|field| field.ident.clone().unwrap())
        .collect();
    let field_tys: Vec<Type> = fields.iter().map(|field| field.ty.clone()).collect();
    let field_indices = (0_u16..).take(field_idents.len());

    Ok(quote! {
        impl #ecs_path::__private::BundleInfo for #struct_name {
            fn component_ids(
                component_manager: &#ecs_path::__private::ComponentManager,
            ) -> Result<
                Vec<#ecs_path::__private::ComponentId>,
                #ecs_path::__private::BundleError
            > {
                let _ = component_manager;

                Ok(vec![
                    #(
                        component_manager
                            .id_of::<#field_tys>()
                            .ok_or(
                                #ecs_path::__private::BundleError::ComponentNotRegistered(
                                    ::core::any::TypeId::of::<#field_tys>()
                                )
                            )?,
                    )*
                ])
            }
        }
        // SAFETY: every field is wrapped in `ManuallyDrop`, converted using its
        // concrete component type, and processed by the cleanup guard exactly
        // once. The generated implementation never accesses a field afterwards.
        unsafe impl #ecs_path::__private::Bundle for #struct_name {
            fn with_items<R, F>(
                self,
                component_manager: &mut #ecs_path::__private::ComponentManager,
                f: F,
            ) -> Result<R, #ecs_path::__private::BundleError>
            where
                F: FnOnce(
                    &mut #ecs_path::__private::ComponentManager,
                    &mut [#ecs_path::__private::ErasedBundleItem]
                ) -> Result<R, #ecs_path::__private::BundleError>,
            {
                let #struct_name {
                    #(#field_idents),*
                } = self;

                #(
                    let mut #field_idents = ::core::mem::ManuallyDrop::new(#field_idents);
                )*

                let mut items = [
                    #(
                        // SAFETY: this field contains a live value at a stable
                        // address. `__bundle_run` either moves or drops it, and
                        // the generated implementation never accesses it again.
                        unsafe {
                            #ecs_path::__private::__bundle_erased_item_of::<#field_tys>(
                                &mut #field_idents
                            )
                        }
                    ),*
                ];

                #ecs_path::__private::__bundle_run(component_manager, &mut items, f)

            }

            unsafe fn __batch_item(
                &mut self,
                item_index: u16,
            ) -> #ecs_path::__private::ErasedBundleItem {
                match item_index {
                    #(
                        #field_indices => unsafe {
                            #ecs_path::__private::__bundle_erased_item_from_ref::<#field_tys>(
                                &mut self.#field_idents
                            )
                        },
                    )*
                    _ => panic!("bundle item index out of bounds"),
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{parse_quote, DeriveInput};

    fn expanded(input: DeriveInput) -> String {
        expand_bundle(input).unwrap().to_string()
    }

    #[test]
    fn named_struct_generates_bundle_impl() {
        let input: DeriveInput = parse_quote! {
            struct PlayerBundle {
                position: Position,
                velocity: Velocity,
            }
        };

        let output = expanded(input);

        assert!(output.contains("unsafe impl :: millarecs :: __private :: Bundle for PlayerBundle"));
        assert!(output.contains("fn with_items < R , F >"));
        assert!(output.contains("__bundle_run (component_manager , & mut items , f)"));
    }

    #[test]
    fn fields_are_destructured_before_erasure() {
        let input: DeriveInput = parse_quote! {
            struct SpawnBundle {
                health: Health,
                player: Player,
            }
        };

        let output = expanded(input);

        assert!(output.contains("let SpawnBundle { health , player } = self"));
        assert!(output.contains("ManuallyDrop :: new (health)"));
        assert!(output.contains("ManuallyDrop :: new (player)"));
    }

    #[test]
    fn every_field_is_converted_to_an_erased_bundle_item() {
        let input: DeriveInput = parse_quote! {
            struct SpawnBundle {
                health: Health,
                player: Player,
            }
        };

        let output = expanded(input);

        assert!(output.contains("__bundle_erased_item_of :: < Health >"));
        assert!(output.contains("__bundle_erased_item_of :: < Player >"));
        assert!(output.contains("& mut health"));
        assert!(output.contains("& mut player"));
        assert!(output.contains("fn __batch_item"));
        assert!(output.contains("__bundle_erased_item_from_ref :: < Health >"));
        assert!(output.contains("__bundle_erased_item_from_ref :: < Player >"));
    }

    #[test]
    fn tuple_struct_is_rejected() {
        let input: DeriveInput = parse_quote! {
            struct InvalidBundle(Position, Velocity);
        };

        let error = expand_bundle(input).unwrap_err().to_string();

        assert_eq!(error, "Bundle require named fields");
    }

    #[test]
    fn unit_struct_is_rejected() {
        let input: DeriveInput = parse_quote! {
            struct InvalidBundle;
        };

        let error = expand_bundle(input).unwrap_err().to_string();

        assert_eq!(error, "Bundle require named fields");
    }

    #[test]
    fn enum_is_rejected() {
        let input: DeriveInput = parse_quote! {
            enum InvalidBundle {
                A,
            }
        };

        let error = expand_bundle(input).unwrap_err().to_string();

        assert_eq!(error, "Bundle can only derive from structs");
    }

    #[test]
    fn union_is_rejected() {
        let input: DeriveInput = parse_quote! {
            union InvalidBundle {
                a: u32,
                b: f32,
            }
        };

        let error = expand_bundle(input).unwrap_err().to_string();

        assert_eq!(error, "Bundle can only derive from structs");
    }

    #[test]
    fn field_types_keep_generic_arguments() {
        let input: DeriveInput = parse_quote! {
            struct GenericBundle {
                values: Vec<Position>,
                marker: Option<Player>,
            }
        };

        let output = expanded(input);

        assert!(output.contains("__bundle_erased_item_of :: < Vec < Position > >"));
        assert!(output.contains("__bundle_erased_item_of :: < Option < Player > >"));
    }
}
