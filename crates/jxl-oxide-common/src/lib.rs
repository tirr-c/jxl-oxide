use jxl_bitstream::Bitstream;

#[macro_export]
#[doc(hidden)]
macro_rules! expand_u32 {
    ($bitstream:ident; $($rest:tt)*) => {
        $crate::expand_u32!(@convert $bitstream; (); ($($rest)*,))
    };
    (@convert $bitstream:ident; ($($done:tt)*); ()) => {
        $bitstream.read_u32($($done)*)
    };
    (@convert $bitstream:ident; ($($done:tt)*); ($c:literal, $($rest:tt)*)) => {
        $crate::expand_u32!(@convert $bitstream; ($($done)* $c,); ($($rest)*))
    };
    (@convert $bitstream:ident; ($($done:tt)*); (u($n:literal), $($rest:tt)*)) => {
        $crate::expand_u32!(@convert $bitstream; ($($done)* ::jxl_bitstream::U($n),); ($($rest)*))
    };
    (@convert $bitstream:ident; ($($done:tt)*); ($c:literal + u($n:literal), $($rest:tt)*)) => {
        $crate::expand_u32!(@convert $bitstream; ($($done)* $c + ::jxl_bitstream::U($n),); ($($rest)*))
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! read_bits {
    ($bistream:ident, $c:literal $(, $ctx:expr)?) => {
        ::jxl_bitstream::Result::Ok($c)
    };
    ($bitstream:ident, u($n:literal) $(, $ctx:expr)?) => {
        $bitstream.read_bits($n)
    };
    ($bitstream:ident, u($n:literal); UnpackSigned $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map(::jxl_bitstream::unpack_signed)
    };
    ($bitstream:ident, $c:literal + u($n:literal) $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map(|v| v.wrapping_add($c))
    };
    ($bitstream:ident, $c:literal + u($n:literal); UnpackSigned $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map(|v| ::jxl_bitstream::unpack_signed(v.wrapping_add($c)))
    };
    ($bitstream:ident, U32($($args:tt)+) $(, $ctx:expr)?) => {
        $crate::expand_u32!($bitstream; $($args)+)
    };
    ($bitstream:ident, U32($($args:tt)+); UnpackSigned $(, $ctx:expr)?) => {
        $crate::expand_u32!($bitstream; $($args)+).map(::jxl_bitstream::unpack_signed)
    };
    ($bitstream:ident, U64 $(, $ctx:expr)?) => {
        $bitstream.read_u64()
    };
    ($bitstream:ident, U64; UnpackSigned $(, $ctx:expr)?) => {
        $bitstream.read_u64().map(::jxl_bitstream::unpack_signed_u64)
    };
    ($bitstream:ident, F16 $(, $ctx:expr)?) => {
        $bitstream.read_f16_as_f32()
    };
    ($bitstream:ident, Bool $(, $ctx:expr)?) => {
        $bitstream.read_bool()
    };
    ($bitstream:ident, Enum($enumtype:ty) $(, $ctx:expr)?) => {
        $bitstream.read_enum::<$enumtype>()
    };
    ($bitstream:ident, ZeroPadToByte $(, $ctx:expr)?) => {
        $bitstream.zero_pad_to_byte()
    };
    ($bitstream:ident, Bundle($bundle:ty)) => {
        <$bundle>::parse($bitstream, ())
    };
    ($bitstream:ident, Bundle($bundle:ty), $ctx:expr) => {
        <$bundle>::parse($bitstream, $ctx)
    };
    ($bitstream:ident, Vec[$($inner:tt)*]; $count:expr $(, $ctx:expr)?) => {
        {
            let count = $count as usize;
            (0..count)
                .into_iter()
                .map(|_| $crate::read_bits!($bitstream, $($inner)* $(, $ctx)?))
                .collect::<::std::result::Result<Vec<_>, _>>()
        }
    };
    ($bitstream:ident, Array[$($inner:tt)*]; $count:expr $(, $ctx:expr)?) => {
        (|| -> ::std::result::Result<[_; $count], _> {
            let mut ret = [Default::default(); $count];
            for point in &mut ret {
                *point = match $crate::read_bits!($bitstream, $($inner)* $(, $ctx)?) {
                    ::std::result::Result::Ok(v) => v,
                    ::std::result::Result::Err(err) => return ::std::result::Result::Err(err),
                };
            }
            Ok(ret)
        })()
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! make_def {
    (@ty; $c:literal) => { u32 };
    (@ty; u($n:literal)) => { u32 };
    (@ty; u($n:literal); UnpackSigned) => { i32 };
    (@ty; $c:literal + u($n:literal)) => { u32 };
    (@ty; $c:literal + u($n:literal); UnpackSigned) => { i32 };
    (@ty; U32($($args:tt)*)) => { u32 };
    (@ty; U32($($args:tt)*); UnpackSigned) => { i32 };
    (@ty; U64) => { u64 };
    (@ty; U64; UnpackSigned) => { i64 };
    (@ty; F16) => { f32 };
    (@ty; Bool) => { bool };
    (@ty; Enum($enum:ty)) => { $enum };
    (@ty; Bundle($bundle:ty)) => { $bundle };
    (@ty; Vec[$($inner:tt)*]; $count:expr) => { Vec<$crate::make_def!(@ty; $($inner)*)> };
    (@ty; Array[$($inner:tt)*]; $count:expr) => { [$crate::make_def!(@ty; $($inner)*); $count] };
    ($(#[$attrs:meta])* $v:vis struct $bundle_name:ident {
        $($(#[$fieldattrs:meta])* $vfield:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        $(#[$attrs])*
        $v struct $bundle_name {
            $($(#[$fieldattrs])* $vfield $field: $crate::make_def!(@ty; $($expr)*),)*
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! make_parse {
    (@parse $bitstream:ident; cond($cond:expr); default($def_expr:expr); ty($($spec:tt)*); ctx($ctx:expr)) => {
        if $cond {
            $crate::read_bits!($bitstream, $($spec)*, $ctx)?
        } else {
            $def_expr
        }
    };
    (@parse $bitstream:ident; cond($cond:expr); ty($($spec:tt)*); ctx($ctx:expr)) => {
        if $cond {
            $crate::read_bits!($bitstream, $($spec)*, $ctx)?
        } else {
            <$crate::make_def!(@ty; $($spec)*)>::default_with_context($ctx)
        }
    };
    (@parse $bitstream:ident; $(default($def_expr:expr);)? ty($($spec:tt)*); ctx($ctx:expr)) => {
        $crate::read_bits!($bitstream, $($spec)*, $ctx)?
    };
    (@default($($spec:tt)*); ; $ctx:expr) => {
        <$crate::make_def!(@ty; $($spec)*)>::default_with_context($ctx)
    };
    (@default($($spec:tt)*); $def_expr:expr $(; $ctx:expr)?) => {
        $def_expr
    };
    (@select_ctx; $ctx_id:ident; $ctx:expr) => {
        $ctx
    };
    (@select_ctx; $ctx_id:ident;) => {
        $ctx_id
    };
    (@select_error_ty;) => {
        ::jxl_bitstream::Error
    };
    (@select_error_ty; $err:ty) => {
        $err
    };
    ($bundle_name:ident $(error($err:ty))? {
        $($(#[$fieldattrs:meta])* $v:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl<Ctx: Copy> $crate::Bundle<Ctx> for $bundle_name {
            type Error = $crate::make_parse!(@select_error_ty; $($err)?);

            #[allow(unused)]
            fn parse(bitstream: &mut ::jxl_bitstream::Bitstream, ctx: Ctx) -> ::std::result::Result<Self, Self::Error> where Self: Sized {
                use $crate::{Bundle, BundleDefault};
                $(
                    let $field = $crate::make_parse!(
                        @parse bitstream;
                        $(cond($cond);)?
                        $(default($def_expr);)?
                        ty($($expr)*);
                        ctx($crate::make_parse!(@select_ctx; ctx; $($ctx_for_field)?))
                    );
                )*
                Ok(Self { $($field,)* })
            }
        }

        impl<Ctx: Copy> $crate::BundleDefault<Ctx> for $bundle_name {
            #[allow(unused)]
            fn default_with_context(_ctx: Ctx) -> Self where Self: Sized {
                use $crate::BundleDefault;
                $(
                    let $field = $crate::make_parse!(
                        @default($($expr)*);
                        $($def_expr)?;
                        $crate::make_parse!(@select_ctx; _ctx; $($ctx_for_field)?)
                    );
                )*
                Self { $($field,)* }
            }
        }
    };
    ($bundle_name:ident ctx($ctx_id:ident : $ctx:ty) $(error($err:ty))? {
        $($(#[$fieldattrs:meta])* $v:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl $crate::Bundle<$ctx> for $bundle_name {
            type Error = $crate::make_parse!(@select_error_ty; $($err)?);

            #[allow(unused)]
            fn parse(bitstream: &mut ::jxl_bitstream::Bitstream, $ctx_id: $ctx) -> ::std::result::Result<Self, Self::Error> where Self: Sized {
                use $crate::{Bundle, BundleDefault};
                $(
                    let $field = $crate::make_parse!(
                        @parse bitstream;
                        $(cond($cond);)?
                        $(default($def_expr);)?
                        ty($($expr)*);
                        ctx($crate::make_parse!(@select_ctx; $ctx_id; $($ctx_for_field)?))
                    );
                )*
                Ok(Self { $($field,)* })
            }
        }

        impl $crate::BundleDefault<$ctx> for $bundle_name {
            #[allow(unused)]
            fn default_with_context($ctx_id: $ctx) -> Self where Self: Sized {
                use $crate::BundleDefault;
                $(
                    let $field = $crate::make_parse!(
                        @default($($expr)*);
                        $($def_expr)?;
                        $crate::make_parse!(@select_ctx; $ctx_id; $($ctx_for_field)?)
                    );
                )*
                Self { $($field,)* }
            }
        }
    };
}

#[macro_export]
macro_rules! define_bundle {
    (
        $(
            $(#[$attrs:meta])*
            $v:vis struct $bundle_name:ident
            $(aligned($aligned:literal))?
            $(ctx($ctx_id:ident : $ctx:ty))?
            $(error($err:ty))?
            {
                $($body:tt)*
            }
        )*
    ) => {
        $(
            $crate::make_def!($(#[$attrs])* $v struct $bundle_name { $($body)* });
            $crate::make_parse!($bundle_name $(aligned($aligned))? $(ctx($ctx_id: $ctx))? $(error($err))? { $($body)* });
        )*
    };
}

pub trait Bundle<Ctx = ()>: Sized {
    type Error;

    /// Parses a value from the bitstream with the given context.
    fn parse(bitstream: &mut Bitstream<'_>, ctx: Ctx) -> Result<Self, Self::Error>;
}

pub trait BundleDefault<Ctx = ()>: Sized {
    /// Creates a default value with the given context.
    fn default_with_context(ctx: Ctx) -> Self;
}

impl<T, Ctx> BundleDefault<Ctx> for T
where
    T: Default + Sized,
{
    fn default_with_context(_: Ctx) -> Self {
        Default::default()
    }
}

impl<T, Ctx> Bundle<Ctx> for Option<T>
where
    T: Bundle<Ctx>,
{
    type Error = T::Error;

    fn parse(bitstream: &mut Bitstream, ctx: Ctx) -> Result<Self, Self::Error> {
        T::parse(bitstream, ctx).map(Some)
    }
}

/// Name type which is read by some JPEG XL headers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Name(String);

impl<Ctx> Bundle<Ctx> for Name {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: Ctx) -> Result<Self, Self::Error> {
        let len = read_bits!(bitstream, U32(0, u(4), 16 + u(5), 48 + u(10)))? as usize;
        let mut data = vec![0u8; len];
        for b in &mut data {
            *b = bitstream.read_bits(8)? as u8;
        }
        let name = String::from_utf8(data)
            .map_err(|_| jxl_bitstream::Error::ValidationFailed("non-UTF-8 name"))?;
        Ok(Self(name))
    }
}

impl std::ops::Deref for Name {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Name {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
