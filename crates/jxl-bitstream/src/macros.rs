#[macro_export]
macro_rules! expand_u32 {
    ($bitstream:ident; $($rest:tt)*) => {
        $bitstream.read_bits(2)
            .and_then(|selector| $crate::expand_u32!(@expand $bitstream, selector, 0; $($rest)*,))
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr;) => {
        unreachable!()
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; $c:literal, $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, $c)
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; u($n:literal), $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, u($n))
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; $c:literal + u($n:literal), $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, $c + u($n))
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
}

#[macro_export]
macro_rules! read_bits {
    ($bistream:ident, $c:literal $(, $ctx:expr)?) => {
        $crate::Result::Ok($c)
    };
    ($bitstream:ident, u($n:literal) $(, $ctx:expr)?) => {
        $bitstream.read_bits($n)
    };
    ($bitstream:ident, u($n:literal); UnpackSigned $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map($crate::unpack_signed)
    };
    ($bitstream:ident, $c:literal + u($n:literal) $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map(|v| v.wrapping_add($c))
    };
    ($bitstream:ident, $c:literal + u($n:literal); UnpackSigned $(, $ctx:expr)?) => {
        $bitstream.read_bits($n).map(|v| $crate::unpack_signed(v.wrapping_add($c)))
    };
    ($bitstream:ident, U32($($args:tt)+) $(, $ctx:expr)?) => {
        $crate::expand_u32!($bitstream; $($args)+)
    };
    ($bitstream:ident, U32($($args:tt)+); UnpackSigned $(, $ctx:expr)?) => {
        $crate::expand_u32!($bitstream; $($args)+).map($crate::unpack_signed)
    };
    ($bitstream:ident, U64 $(, $ctx:expr)?) => {
        $bitstream.read_u64()
    };
    ($bitstream:ident, U64; UnpackSigned $(, $ctx:expr)?) => {
        read_bits!($bitstream, U64 $(, $ctx)?).map($crate::unpack_signed_u64)
    };
    ($bitstream:ident, F16 $(, $ctx:expr)?) => {
        $bitstream.read_f16_as_f32()
    };
    ($bitstream:ident, Bool $(, $ctx:expr)?) => {
        $bitstream.read_bool()
    };
    ($bitstream:ident, Enum($enumtype:ty) $(, $ctx:expr)?) => {
        $crate::read_bits!($bitstream, U32(0, 1, 2 + u(4), 18 + u(6)))
            .and_then(|v| {
                <$enumtype as TryFrom<u32>>::try_from(v).map_err(|_| $crate::Error::InvalidEnum {
                    name: stringify!($enumtype),
                    value: v,
                })
            })
    };
    ($bitstream:ident, ZeroPadToByte $(, $ctx:expr)?) => {
        $bitstream.zero_pad_to_byte()
    };
    ($bitstream:ident, Bundle($bundle:ty)) => {
        $bitstream.read_bundle::<$bundle>()
    };
    ($bitstream:ident, Bundle($bundle:ty), $ctx:expr) => {
        $bitstream.read_bundle_with_ctx::<$bundle, _>($ctx)
    };
    ($bitstream:ident, Vec[$($inner:tt)*]; $count:expr $(, $ctx:expr)?) => {
        {
            let count = $count as usize;
            (0..count)
                .into_iter()
                .map(|_| $crate::read_bits!($bitstream, $($inner)* $(, $ctx)?))
                .collect::<$crate::Result<Vec<_>>>()
        }
    };
    ($bitstream:ident, Array[$($inner:tt)*]; $count:expr $(, $ctx:expr)?) => {
        (|| -> $crate::Result<[_; $count]> {
            let mut ret = [Default::default(); $count];
            for point in &mut ret {
                *point = $crate::read_bits!($bitstream, $($inner)* $(, $ctx)?)?;
            }
            Ok(ret)
        })()
    };
}

#[macro_export]
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
        $($vfield:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        $(#[$attrs])*
        $v struct $bundle_name {
            $($vfield $field: $crate::make_def!(@ty; $($expr)*),)*
        }
    };
}

#[macro_export]
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
            $crate::BundleDefault::default_with_context($ctx)
        }
    };
    (@parse $bitstream:ident; $(default($def_expr:expr);)? ty($($spec:tt)*); ctx($ctx:expr)) => {
        $crate::read_bits!($bitstream, $($spec)*, $ctx)?
    };
    (@default; ; $ctx:expr) => {
        $crate::BundleDefault::default_with_context($ctx)
    };
    (@default; $def_expr:expr $(; $ctx:expr)?) => {
        $def_expr
    };
    (@select_ctx; $ctx_id:ident; $ctx:expr) => {
        $ctx
    };
    (@select_ctx; $ctx_id:ident;) => {
        $ctx_id
    };
    ($bundle_name:ident {
        $($v:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl<Ctx: Copy> $crate::Bundle<Ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn parse<R: ::std::io::Read>(bitstream: &mut $crate::Bitstream<R>, ctx: Ctx) -> $crate::Result<Self> where Self: Sized {
                $(
                    let $field: $crate::make_def!(@ty; $($expr)*) = $crate::make_parse!(
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
            #[allow(unused_variables)]
            fn default_with_context(_ctx: Ctx) -> Self where Self: Sized {
                $(
                    let $field: $crate::make_def!(@ty; $($expr)*) = $crate::make_parse!(
                        @default;
                        $($def_expr)?;
                        $crate::make_parse!(@select_ctx; _ctx; $($ctx_for_field)?)
                    );
                )*
                Self { $($field,)* }
            }
        }
    };
    ($bundle_name:ident ctx($ctx_id:ident : $ctx:ty) {
        $($v:vis $field:ident: ty($($expr:tt)*) $(ctx($ctx_for_field:expr))? $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl $crate::Bundle<$ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn parse<R: ::std::io::Read>(bitstream: &mut $crate::Bitstream<R>, $ctx_id: $ctx) -> $crate::Result<Self> where Self: Sized {
                $(
                    let $field: $crate::make_def!(@ty; $($expr)*) = $crate::make_parse!(
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
            #[allow(unused_variables)]
            fn default_with_context($ctx_id: $ctx) -> Self where Self: Sized {
                $(
                    let $field: $crate::make_def!(@ty; $($expr)*) = $crate::make_parse!(
                        @default;
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
            $(#[$attrs:meta])* $v:vis struct $bundle_name:ident $(aligned($aligned:literal))? $(ctx($ctx_id:ident : $ctx:ty))? { $($body:tt)* }
        )*
    ) => {
        $(
            $crate::make_def!($(#[$attrs])* $v struct $bundle_name { $($body)* });
            $crate::make_parse!($bundle_name $(aligned($aligned))? $(ctx($ctx_id: $ctx))? { $($body)* });
        )*
    };
}

pub fn unpack_signed(x: u32) -> i32 {
    let base = (x >> 1) as i32;
    if x & 1 == 0 {
        base
    } else {
        -base - 1
    }
}

pub fn unpack_signed_u64(x: u64) -> i64 {
    let base = (x >> 1) as i64;
    if x & 1 == 0 {
        base
    } else {
        -base - 1
    }
}
