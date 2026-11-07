//! Tuple-arity code generation helper.
//!
//! `all_tuples!` invokes another macro for each arity in an inclusive range,
//! avoiding hand-written implementations for repeated tuple trait patterns.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    token::Comma,
    Error, Ident, LitInt, Path, Result,
};

/// Parsed input for `all_tuples!`.
///
/// The macro accepts a target macro path, an inclusive arity range, and one or
/// more identifier prefixes used to build generated identifiers such as `T0`,
/// `T1`, and `T2`.
struct ParsedTuplesData {
    macro_path: Path,
    min_arity: usize,
    max_arity: usize,
    /// Order matters: at each arity position the prefixes are visited in
    /// declaration order, interleaving their identifiers.
    prefixes: Vec<Ident>,
}

impl Parse for ParsedTuplesData {
    /// Parses the macro path, arity range, and prefix list, rejecting empty
    /// prefix lists and inverted ranges.
    fn parse(input: ParseStream) -> Result<Self> {
        let macro_path = input.parse::<Path>()?;
        input.parse::<Comma>()?;

        let min_arity = input.parse::<LitInt>()?.base10_parse::<usize>()?;
        input.parse::<Comma>()?;

        let max_arity_span = input.span();
        let max_arity = input.parse::<LitInt>()?.base10_parse::<usize>()?;
        input.parse::<Comma>()?;

        let prefixes = Punctuated::<Ident, Comma>::parse_terminated(input)?
            .into_iter()
            .collect::<Vec<_>>();

        if prefixes.is_empty() {
            return Err(Error::new(
                input.span(),
                "expected at least one identifier prefix",
            ));
        }

        if max_arity < min_arity {
            return Err(Error::new(
                max_arity_span,
                "max_arity should be higher than min_arity",
            ));
        }

        Ok(ParsedTuplesData {
            macro_path,
            min_arity,
            max_arity,
            prefixes,
        })
    }
}

/// Expands the parsed tuple range into repeated macro invocations.
pub fn all_tuples(input: TokenStream) -> TokenStream {
    let data = parse_macro_input!(input as ParsedTuplesData);
    TokenStream::from(expand_all_tuples(data))
}

/// Emits one target macro invocation per arity in the inclusive range.
fn expand_all_tuples(data: ParsedTuplesData) -> TokenStream2 {
    let ParsedTuplesData {
        macro_path,
        min_arity,
        max_arity,
        prefixes,
    } = data;

    let invocations = (min_arity..=max_arity).map(|arity| {
        let ident_list = build_ident_list(&prefixes, arity);

        quote! {
            #macro_path!(#[allow(non_snake_case)] #ident_list);
        }
    });

    quote! {
        #(
            #invocations
        )*
    }
}

/// Builds the comma-separated identifier list for a single arity.
///
/// Prefixes interleave per arity position in declaration order, so `S, M`
/// with arity three yields `S0, M0, S1, M1, S2, M2`.
fn build_ident_list(prefixes: &[Ident], arity: usize) -> TokenStream2 {
    let idents = (0..arity).flat_map(|n| {
        prefixes
            .iter()
            .map(move |prefix| format_ident!("{}{}", prefix, n))
    });

    quote! {
        #(#idents),*
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    fn parse_data(tokens: TokenStream2) -> syn::Result<ParsedTuplesData> {
        syn::parse2(tokens)
    }

    #[test]
    fn parses_macro_path_range_and_prefix() {
        let data = parse_data(quote!(impl_tuple, 1, 4, T)).unwrap();

        assert_eq!(
            data.macro_path.segments.last().unwrap().ident.to_string(),
            "impl_tuple"
        );
        assert_eq!(data.min_arity, 1);
        assert_eq!(data.max_arity, 4);
        assert_eq!(data.prefixes.len(), 1);
        assert_eq!(data.prefixes[0].to_string(), "T");
    }

    #[test]
    fn parses_multiple_prefixes() {
        let data = parse_data(quote!(impl_tuple, 1, 4, S, M)).unwrap();

        assert_eq!(data.prefixes.len(), 2);
        assert_eq!(data.prefixes[0].to_string(), "S");
        assert_eq!(data.prefixes[1].to_string(), "M");
    }

    #[test]
    fn parses_nested_macro_paths() {
        let data = parse_data(quote!(crate::inner::impl_tuple, 0, 2, P)).unwrap();

        assert_eq!(data.macro_path.segments.len(), 3);
        assert_eq!(data.prefixes[0].to_string(), "P");
    }

    #[test]
    fn rejects_missing_identifier_prefixes() {
        let error = match parse_data(quote!(impl_tuple, 0, 2,)) {
            Ok(_) => panic!("expected missing prefix error"),
            Err(error) => error,
        };

        assert_eq!(error.to_string(), "expected at least one identifier prefix");
    }

    #[test]
    fn rejects_ranges_where_max_is_lower_than_min() {
        let error = match parse_data(quote!(impl_tuple, 3, 2, T)) {
            Ok(_) => panic!("expected invalid range error"),
            Err(error) => error,
        };

        assert_eq!(
            error.to_string(),
            "max_arity should be higher than min_arity"
        );
    }

    #[test]
    fn build_ident_list_supports_zero_arity() {
        let idents = vec![parse_quote!(T0), parse_quote!(T1)];

        let output = build_ident_list(&idents, 0).to_string();

        assert_eq!(output, "");
    }

    #[test]
    fn build_ident_list_uses_the_requested_prefix_order() {
        let prefixes = vec![parse_quote!(P)];

        let output = build_ident_list(&prefixes, 3).to_string();

        assert_eq!(output, "P0 , P1 , P2");
    }

    #[test]
    fn build_ident_list_supports_multiple_prefixes() {
        let prefixes = vec![parse_quote!(S), parse_quote!(M)];

        let output = build_ident_list(&prefixes, 3).to_string();

        assert_eq!(output, "S0 , M0 , S1 , M1 , S2 , M2");
    }

    #[test]
    fn expansion_invokes_target_macro_for_each_arity() {
        let data = parse_data(quote!(impl_tuple, 0, 3, T)).unwrap();

        let output = expand_all_tuples(data).to_string();

        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)])"));
        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)] T0)"));
        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)] T0 , T1)"));
        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)] T0 , T1 , T2)"));
    }

    #[test]
    fn expansion_can_start_from_non_zero_arity() {
        let data = parse_data(quote!(impl_tuple, 2, 3, T)).unwrap();

        let output = expand_all_tuples(data).to_string();

        assert!(!output.contains("impl_tuple ! (# [allow (non_snake_case)])"));
        assert!(!output.contains("impl_tuple ! (# [allow (non_snake_case)] T0)"));
        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)] T0 , T1)"));
        assert!(output.contains("impl_tuple ! (# [allow (non_snake_case)] T0 , T1 , T2)"));
    }

    #[test]
    fn expansion_keeps_nested_macro_paths() {
        let data = parse_data(quote!(crate::macros::impl_tuple, 1, 1, A)).unwrap();

        let output = expand_all_tuples(data).to_string();

        assert!(output.contains("crate :: macros :: impl_tuple"));
        assert!(output.contains("A0"));
    }
}
