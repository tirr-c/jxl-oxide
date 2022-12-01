#![allow(dead_code, clippy::excessive_precision)]
use crate::{Bitstream, Bundle};

macro_rules! make_def {
    (@ty; $c:literal) => { u32 };
    (@ty; u($n:literal)) => { u32 };
    (@ty; $c:literal + u($n:literal)) => { u32 };
    (@ty; U32($($args:tt)*)) => { u32 };
    (@ty; U64) => { u64 };
    (@ty; F16) => { f32 };
    (@ty; Bool) => { bool };
    (@ty; Enum($enum:ty)) => { $enum };
    (@ty; Bundle($bundle:ty)) => { $bundle };
    (@ty; Vec[$($inner:tt)*]; $count:expr) => { Vec<make_def!(@ty; $($inner)*)> };
    (@ty; Array[$($inner:tt)*]; $count:expr) => { [make_def!(@ty; $($inner)*); $count] };
    ($(#[$attrs:meta])* $v:vis struct $bundle_name:ident {
        $($vfield:vis $field:ident: ty($($expr:tt)*) $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        $(#[$attrs])*
        $v struct $bundle_name {
            $($vfield $field: make_def!(@ty; $($expr)*),)*
        }
    };
}

macro_rules! make_parse {
    (@parse $bitstream:ident; cond($cond:expr); default($def_expr:expr); ty($($spec:tt)*) ctx($ctx:ident)) => {
        if $cond {
            $crate::read_bits!($bitstream, $($spec)*, $ctx)?
        } else {
            $def_expr
        }
    };
    (@parse $bitstream:ident; cond($cond:expr); ty($($spec:tt)*) ctx($ctx:ident)) => {
        if $cond {
            $crate::read_bits!($bitstream, $($spec)*, $ctx)?
        } else {
            $crate::BundleDefault::default_with_context($ctx)
        }
    };
    (@parse $bitstream:ident; $(default($def_expr:expr);)? ty($($spec:tt)*) ctx($ctx:ident)) => {
        $crate::read_bits!($bitstream, $($spec)*, $ctx)?
    };
    (@default; ; $ctx:ident) => {
        $crate::BundleDefault::default_with_context($ctx)
    };
    (@default; $def_expr:expr $(; $ctx:ident)?) => {
        $def_expr
    };
    ($bundle_name:ident {
        $($v:vis $field:ident: ty($($expr:tt)*) $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl<Ctx> $crate::Bundle<Ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
                $(
                    let $field = make_parse!(@parse bitstream; $(cond($cond);)? $(default($def_expr);)? ty($($expr)*) ctx(ctx));
                )*
                Ok(Self { $($field,)* })
            }
        }

        impl<Ctx> $crate::BundleDefault<Ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn default_with_context(_ctx: &Ctx) -> Self where Self: Sized {
                $(
                    let $field = make_parse!(@default; $($def_expr)?; _ctx);
                )*
                Self { $($field,)* }
            }
        }
    };
    ($bundle_name:ident ctx($ctx_id:ident : $ctx:ty) {
        $($v:vis $field:ident: ty($($expr:tt)*) $(cond($cond:expr))? $(default($def_expr:expr))? ,)*
    }) => {
        impl $crate::Bundle<$ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, $ctx_id: &$ctx) -> crate::Result<Self> where Self: Sized {
                $(
                    let $field = make_parse!(@parse bitstream; $(cond($cond);)? $(default($def_expr);)? ty($($expr)*) ctx($ctx_id));
                )*
                Ok(Self { $($field,)* })
            }
        }

        impl $crate::BundleDefault<$ctx> for $bundle_name {
            #[allow(unused_variables)]
            fn default_with_context($ctx_id: &$ctx) -> Self where Self: Sized {
                $(
                    let $field = make_parse!(@default; $($def_expr)?; $ctx_id);
                )*
                Self { $($field,)* }
            }
        }
    };
}

macro_rules! define_bundle {
    (
        $(
            $(#[$attrs:meta])* $v:vis struct $bundle_name:ident $(aligned($aligned:literal))? $(ctx($ctx_id:ident : $ctx:ty))? { $($body:tt)* }
        )*
    ) => {
        $(
            make_def!($(#[$attrs])* $v struct $bundle_name { $($body)* });
            make_parse!($bundle_name $(aligned($aligned))? $(ctx($ctx_id: $ctx))? { $($body)* });
        )*
    };
}

define_bundle! {
    #[derive(Debug)]
    pub struct Headers {
        pub signature: ty(u(16)),
        pub size: ty(Bundle(SizeHeader)),
        pub metadata: ty(Bundle(ImageMetadata)),
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct SizeHeader {
        div8: ty(Bool) default(false),
        h_div8: ty(1 + u(5)) cond(div8) default(0),
        pub height:
            ty(U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))) cond(!div8)
            default(8 * h_div8),
        ratio: ty(u(3)) default(0),
        w_div8: ty(1 + u(5)) cond(div8 && ratio == 0) default(0),
        pub width:
            ty(U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))) cond(!div8 && ratio == 0)
            default(SizeHeader::compute_default_width(ratio, w_div8, height)),
    }
}

impl SizeHeader {
    fn compute_default_width(ratio: u32, w_div8: u32, height: u32) -> u32 {
        match ratio {
            0 => 8 * w_div8,
            1 => height,
            2 => height * 12 / 10,
            3 => height * 4 / 3,
            4 => height * 3 / 2,
            5 => height * 16 / 9,
            6 => height * 5 / 4,
            7 => height * 2,
            _ => panic!("Invalid ratio const: {}", ratio),
        }
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct ImageMetadata {
        all_default: ty(Bool) default(true),
        pub extra_fields: ty(Bool) cond(!all_default) default(false),
        pub orientation: ty(1 + u(3)) cond(extra_fields) default(1),
        pub have_intr_size: ty(Bool) cond(extra_fields) default(false),
        pub intrinsic_size: ty(Bundle(SizeHeader)) cond(have_intr_size),
        pub have_preview: ty(Bool) cond(extra_fields) default(false),
        pub preview: ty(Bundle(PreviewHeader)) cond(have_preview),
        pub have_animation: ty(Bool) cond(extra_fields) default(false),
        pub animation: ty(Bundle(AnimationHeader)) cond(have_animation),
        pub bit_depth: ty(Bundle(BitDepth)) cond(!all_default),
        pub modular_16bit_buffers: ty(Bool) cond(!all_default) default(true),
        pub num_extra: ty(U32(0, 1, 2 + u(4), 1 + u(12))) cond(!all_default) default(0),
        pub ec_info: ty(Vec[Bundle(ExtraChannelInfo)]; num_extra) cond(!all_default),
        pub xyb_encoded: ty(Bool) cond(!all_default) default(true),
        pub colour_encoding: ty(Bundle(ColourEncoding)) cond(!all_default),
        pub tone_mapping: ty(Bundle(ToneMapping)) cond(extra_fields),
        pub extensions: ty(Bundle(Extensions)) cond(!all_default),
        pub default_m: ty(Bool),
        pub opsin_inverse_matrix: ty(Bundle(OpsinInverseMatrix)) cond(!default_m && xyb_encoded),
        pub cw_mask: ty(u(3)) cond(!default_m) default(0),
        pub up2_weight: ty(Array[F16]; 15) cond(cw_mask & 1 != 0) default(Self::D_UP2),
        pub up4_weight: ty(Array[F16]; 55) cond(cw_mask & 2 != 0) default(Self::D_UP4),
        pub up8_weight: ty(Array[F16]; 210) cond(cw_mask & 4 != 0) default(Self::D_UP8),
    }

    #[derive(Debug)]
    pub struct PreviewHeader {
        div8: ty(Bool),
        h_div8: ty(U32(16, 32, 1 + u(5), 33 + u(9))) cond(div8) default(1),
        pub height:
            ty(U32(1 + u(6), 65 + u(8), 321 + u(10), 1345 + u(12))) cond(!div8)
            default(8 * h_div8),
        ratio: ty(u(3)),
        w_div8: ty(U32(16, 32, 1 + u(5), 33 + u(9))) cond(div8) default(1),
        pub width:
            ty(U32(1 + u(6), 65 + u(8), 321 + u(10), 1345 + u(12))) cond(!div8)
            default(SizeHeader::compute_default_width(ratio, w_div8, height)),
    }

    #[derive(Debug)]
    pub struct AnimationHeader {
        pub tps_numerator: ty(U32(100, 1000, 1 + u(10), 1 + u(30))) default(0),
        pub tps_denominator: ty(U32(1, 1001, 1 + u(8), 1 + u(10))) default(0),
        pub num_loops: ty(U32(0, u(3), u(16), u(32))) default(0),
        pub have_timecodes: ty(Bool) default(false),
    }

    #[derive(Debug)]
    pub struct ExtraChannelInfo {
        pub d_alpha: ty(Bool) default(true),
        pub ty: ty(Enum(ExtraChannelType)) cond(!d_alpha) default(ExtraChannelType::Alpha),
        pub bit_depth: ty(Bundle(BitDepth)) cond(!d_alpha),
        pub dim_shift: ty(U32(0, 3, 4, 1 + u(3))) cond(!d_alpha) default(0),
        pub name_len: ty(U32(0, u(4), 16 + u(5), 48 + u(10))) cond(!d_alpha) default(0),
        pub name: ty(Vec[u(8)]; name_len) default(vec![0; name_len as usize]),
        pub alpha_associated: ty(Bool) cond(!d_alpha && ty == ExtraChannelType::Alpha) default(false),
        pub red: ty(F16) cond(ty == ExtraChannelType::SpotColour) default(0.0),
        pub green: ty(F16) cond(ty == ExtraChannelType::SpotColour) default(0.0),
        pub blue: ty(F16) cond(ty == ExtraChannelType::SpotColour) default(0.0),
        pub solidity: ty(F16) cond(ty == ExtraChannelType::SpotColour) default(0.0),
        pub cfa_channel: ty(U32(1, u(2), 3 + u(4), 19 + u(8))) cond(ty == ExtraChannelType::Cfa) default(1),
    }

    #[derive(Debug)]
    pub struct Extensions {
        extensions: ty(U64) default(0),
        extension_bits:
            ty(Vec[U64]; (extensions + 7) / 8) cond(extensions != 0)
            default(vec![0; ((extensions + 7) / 8) as usize]),
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ExtraChannelType {
    #[default]
    Alpha = 0,
    Depth,
    SpotColour,
    SelectionMask,
    Black,
    Cfa,
    Thermal,
    NonOptional = 15,
    Optional,
}

impl TryFrom<u32> for ExtraChannelType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Alpha,
            1 => Self::Depth,
            2 => Self::SpotColour,
            3 => Self::SelectionMask,
            4 => Self::Black,
            5 => Self::Cfa,
            6 => Self::Thermal,
            15 => Self::NonOptional,
            16 => Self::Optional,
            _ => return Err(()),
        })
    }
}

#[derive(Debug)]
pub enum BitDepth {
    IntegerSample {
        bits_per_sample: u32,
    },
    FloatSample {
        bits_per_sample: u32,
        exp_bits: u32,
    },
}

impl Default for BitDepth {
    fn default() -> Self {
        Self::IntegerSample { bits_per_sample: 8 }
    }
}

impl<Ctx> Bundle<Ctx> for BitDepth {
    fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        if crate::read_bits!(bitstream, Bool)? { // float_sample
            let bits_per_sample = crate::read_bits!(bitstream, U32(32, 16, 24, 1 + u(6)))?;
            let exp_bits = crate::read_bits!(bitstream, 1 + u(4))?;
            Ok(Self::FloatSample { bits_per_sample, exp_bits })
        } else {
            let bits_per_sample = crate::read_bits!(bitstream, U32(8, 10, 12, 1 + u(6)))?;
            Ok(Self::IntegerSample { bits_per_sample })
        }
    }
}

impl ImageMetadata {
    const D_UP2: [f32; 15] = [
        -0.01716200, -0.03452303, -0.04022174, -0.02921014, -0.00624645,
        0.14111091, 0.28896755, 0.00278718, -0.01610267, 0.56661550,
        0.03777607, -0.01986694, -0.03144731, -0.01185068, -0.00213539,
    ];
    const D_UP4: [f32; 55] = [
        0.02419067, -0.03491987, -0.03693351, -0.03094285, -0.00529785,
        -0.01663432, -0.03556863, -0.03888905, -0.03516850, -0.00989469,
        0.23651958, 0.33392945, -0.01073543, -0.01313181, -0.03556694,
        0.13048175, 0.40103025, 0.03951150, -0.02077584, 0.46914198,
        -0.00209270, -0.01484589, -0.04064806, 0.18942530, 0.56279892,
        0.06674400, -0.02335494, -0.03551682, -0.00754830, -0.02267919,
        -0.02363578, 0.00315804, -0.03399098, -0.01359519, -0.00091653,
        -0.00335467, -0.01163294, -0.01610294, -0.00974088, -0.00191622,
        -0.01095446, -0.03198464, -0.04455121, -0.02799790, -0.00645912,
        0.06390599, 0.22963888, 0.00630981, -0.01897349, 0.67537268,
        0.08483369, -0.02534994, -0.02205197, -0.01667999, -0.00384443,
    ];
    const D_UP8: [f32; 210] = [
        -0.02928613, -0.03706353, -0.03783812, -0.03324558, -0.00447632,
        -0.02519406, -0.03752601, -0.03901508, -0.03663285, -0.00646649,
        -0.02066407, -0.03838633, -0.04002101, -0.03900035, -0.00901973,
        -0.01626393, -0.03954148, -0.04046620, -0.03979621, -0.01224485,
        0.29895328, 0.35757708, -0.02447552, -0.01081748, -0.04314594,
        0.23903219, 0.41119301, -0.00573046, -0.01450239, -0.04246845,
        0.17567618, 0.45220643, 0.02287757, -0.01936783, -0.03583255,
        0.11572472, 0.47416733, 0.06284440, -0.02685066, 0.42720050,
        -0.02248939, -0.01155273, -0.04562755, 0.28689496, 0.49093869,
        -0.00007891, -0.01545926, -0.04562659, 0.21238920, 0.53980934,
        0.03369474, -0.02070211, -0.03866988, 0.14229550, 0.56593398,
        0.08045181, -0.02888298, -0.03680918, -0.00542229, -0.02920477,
        -0.02788574, -0.02118180, -0.03942402, -0.00775547, -0.02433614,
        -0.03193943, -0.02030828, -0.04044014, -0.01074016, -0.01930822,
        -0.03620399, -0.01974125, -0.03919545, -0.01456093, -0.00045072,
        -0.00360110, -0.01020207, -0.01231907, -0.00638988, -0.00071592,
        -0.00279122, -0.00957115, -0.01288327, -0.00730937, -0.00107783,
        -0.00210156, -0.00890705, -0.01317668, -0.00813895, -0.00153491,
        -0.02128481, -0.04173044, -0.04831487, -0.03293190, -0.00525260,
        -0.01720322, -0.04052736, -0.05045706, -0.03607317, -0.00738030,
        -0.01341764, -0.03965629, -0.05151616, -0.03814886, -0.01005819,
        0.18968273, 0.33063684, -0.01300105, -0.01372950, -0.04017465,
        0.13727832, 0.36402234, 0.01027890, -0.01832107, -0.03365072,
        0.08734506, 0.38194295, 0.04338228, -0.02525993, 0.56408126,
        0.00458352, -0.01648227, -0.04887868, 0.24585519, 0.62026135,
        0.04314807, -0.02213737, -0.04158014, 0.16637289, 0.65027023,
        0.09621636, -0.03101388, -0.04082742, -0.00904519, -0.02790922,
        -0.02117818, 0.00798662, -0.03995711, -0.01243427, -0.02231705,
        -0.02946266, 0.00992055, -0.03600283, -0.01684920, -0.00111684,
        -0.00411204, -0.01297130, -0.01723725, -0.01022545, -0.00165306,
        -0.00313110, -0.01218016, -0.01763266, -0.01125620, -0.00231663,
        -0.01374149, -0.03797620, -0.05142937, -0.03117307, -0.00581914,
        -0.01064003, -0.03608089, -0.05272168, -0.03375670, -0.00795586,
        0.09628104, 0.27129991, -0.00353779, -0.01734151, -0.03153981,
        0.05686230, 0.28500998, 0.02230594, -0.02374955, 0.68214326,
        0.05018048, -0.02320852, -0.04383616, 0.18459474, 0.71517975,
        0.10805613, -0.03263677, -0.03637639, -0.01394373, -0.02511203,
        -0.01728636, 0.05407331, -0.02867568, -0.01893131, -0.00240854,
        -0.00446511, -0.01636187, -0.02377053, -0.01522848, -0.00333334,
        -0.00819975, -0.02964169, -0.04499287, -0.02745350, -0.00612408,
        0.02727416, 0.19446600, 0.00159832, -0.02232473, 0.74982506,
        0.11452620, -0.03348048, -0.01605681, -0.02070339, -0.00458223,
    ];
}

define_bundle! {
    #[derive(Debug)]
    pub struct ColourEncoding {
        all_default: ty(Bool) default(true),
        pub want_icc: ty(Bool) cond(!all_default) default(false),
        pub colour_space: ty(Enum(ColourSpace)) cond(!all_default) default(ColourSpace::Rgb),
        pub white_point: ty(Bundle(WhitePoint)) cond(!all_default && !want_icc && colour_space != ColourSpace::Xyb) default(WhitePoint::D65),
        pub primaries: ty(Bundle(Primaries)) cond(!all_default && !want_icc && colour_space != ColourSpace::Xyb && colour_space != ColourSpace::Grey) default(Primaries::Srgb),
        pub tf: ty(Bundle(TransferFunction)) cond(!all_default && !want_icc),
        pub rendering_intent: ty(Enum(RenderingIntent)) cond(!all_default && !want_icc) default(RenderingIntent::Relative),
    }

    #[derive(Debug, PartialEq, Eq)]
    pub struct Customxy {
        pub ux: ty(U32(u(19), 524288 + u(19), 1048576 + u(20), 2097152 + u(21))),
        pub uy: ty(U32(u(19), 524288 + u(19), 1048576 + u(20), 2097152 + u(21))),
    }

    #[derive(Debug)]
    pub struct ToneMapping {
        all_default: ty(Bool) default(true),
        pub intensity_target: ty(F16) cond(!all_default) default(255.0),
        pub min_nits: ty(F16) cond(!all_default) default(0.0),
        pub relative_to_max_display: ty(Bool) cond(!all_default) default(false),
        pub linear_below: ty(F16) cond(!all_default) default(0.0),
    }
}

#[derive(Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ColourSpace {
    Rgb = 0,
    Grey = 1,
    Xyb = 2,
    Unknown = 3,
}

impl TryFrom<u32> for ColourSpace {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Rgb,
            1 => Self::Grey,
            2 => Self::Xyb,
            3 => Self::Unknown,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
#[repr(u8)]
enum WhitePointDiscriminator {
    D65 = 1,
    Custom = 2,
    E = 10,
    Dci = 11,
}

impl TryFrom<u32> for WhitePointDiscriminator {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::D65,
            2 => Self::Custom,
            10 => Self::E,
            11 => Self::Dci,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub enum WhitePoint {
    #[default]
    D65,
    Custom(Customxy),
    E,
    Dci,
}

impl<Ctx> Bundle<Ctx> for WhitePoint {
    fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        let d = crate::read_bits!(bitstream, Enum(WhitePointDiscriminator))?;
        Ok(match d {
            WhitePointDiscriminator::D65 => Self::D65,
            WhitePointDiscriminator::E => Self::E,
            WhitePointDiscriminator::Dci => Self::Dci,
            WhitePointDiscriminator::Custom => {
                let white = crate::read_bits!(bitstream, Bundle(Customxy))?;
                Self::Custom(white)
            },
        })
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
enum PrimariesDiscriminator {
    #[default]
    Srgb = 1,
    Custom = 2,
    Bt2100 = 9,
    P3 = 11,
}

impl TryFrom<u32> for PrimariesDiscriminator {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Srgb,
            2 => Self::Custom,
            9 => Self::Bt2100,
            11 => Self::P3,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub enum Primaries {
    #[default]
    Srgb,
    Custom {
        red: Customxy,
        green: Customxy,
        blue: Customxy,
    },
    Bt2100,
    P3,
}

impl<Ctx> Bundle<Ctx> for Primaries {
    fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        let d = crate::read_bits!(bitstream, Enum(PrimariesDiscriminator))?;
        Ok(match d {
            PrimariesDiscriminator::Srgb => Self::Srgb,
            PrimariesDiscriminator::Bt2100 => Self::Bt2100,
            PrimariesDiscriminator::P3 => Self::P3,
            PrimariesDiscriminator::Custom => {
                let red = crate::read_bits!(bitstream, Bundle(Customxy))?;
                let green = crate::read_bits!(bitstream, Bundle(Customxy))?;
                let blue = crate::read_bits!(bitstream, Bundle(Customxy))?;
                Self::Custom { red, green, blue }
            },
        })
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub enum TransferFunction {
    Gamma(u32),
    Bt709,
    Unknown,
    Linear,
    #[default]
    Srgb,
    Pq,
    Dci,
    Hlg,
}

impl TryFrom<u32> for TransferFunction {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Bt709,
            2 => Self::Unknown,
            8 => Self::Linear,
            13 => Self::Srgb,
            16 => Self::Pq,
            17 => Self::Dci,
            18 => Self::Hlg,
            _ => return Err(()),
        })
    }
}

impl<Ctx> Bundle<Ctx> for TransferFunction {
    fn parse<R: ::std::io::Read>(bitstream: &mut Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        let has_gamma = crate::read_bits!(bitstream, Bool)?;
        if has_gamma {
            let gamma = crate::read_bits!(bitstream, u(24))?;
            Ok(Self::Gamma(gamma))
        } else {
            crate::read_bits!(bitstream, Enum(TransferFunction))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RenderingIntent {
    Perceptual = 0,
    #[default]
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

impl TryFrom<u32> for RenderingIntent {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Perceptual,
            1 => Self::Relative,
            2 => Self::Saturation,
            3 => Self::Absolute,
            _ => return Err(()),
        })
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct OpsinInverseMatrix {
        all_default: ty(Bool) default(true),
        inv_mat00: ty(F16) cond(!all_default) default(11.031566901960783),
        inv_mat01: ty(F16) cond(!all_default) default(-9.866943921568629),
        inv_mat02: ty(F16) cond(!all_default) default(-0.16462299647058826),
        inv_mat10: ty(F16) cond(!all_default) default(-3.254147380392157),
        inv_mat11: ty(F16) cond(!all_default) default(4.418770392156863),
        inv_mat12: ty(F16) cond(!all_default) default(-0.16462299647058826),
        inv_mat20: ty(F16) cond(!all_default) default(-3.6588512862745097),
        inv_mat21: ty(F16) cond(!all_default) default(2.7129230470588235),
        inv_mat22: ty(F16) cond(!all_default) default(1.9459282392156863),
        opsin_bias0: ty(F16) cond(!all_default) default(-0.0037930732552754493),
        opsin_bias1: ty(F16) cond(!all_default) default(-0.0037930732552754493),
        opsin_bias2: ty(F16) cond(!all_default) default(-0.0037930732552754493),
        quant_bias0: ty(F16) cond(!all_default) default(1.0 - 0.05465007330715401),
        quant_bias1: ty(F16) cond(!all_default) default(1.0 - 0.07005449891748593),
        quant_bias2: ty(F16) cond(!all_default) default(1.0 - 0.049935103337343655),
        quant_bias_numerator: ty(F16) cond(!all_default) default(0.145),
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct FrameHeader ctx(headers: Headers) {
        all_default: ty(Bool) default(true),
        pub frame_type: ty(Bundle(FrameType)) cond(!all_default) default(FrameType::RegularFrame),
        pub encoding: ty(Bundle(Encoding)) cond(!all_default) default(Encoding::VarDct),
        pub flags: ty(Bundle(FrameFlags)) cond(!all_default),
        pub do_ycbcr: ty(Bool) cond(!all_default && headers.metadata.xyb_encoded),
        pub jpeg_upsampling: ty(Array[u(2)]; 3) cond(do_ycbcr && !flags.use_lf_frame()),
        pub upsampling: ty(U32(1, 2, 4, 8)) cond(do_ycbcr && !flags.use_lf_frame()) default(1),
        pub ec_upsampling:
            ty(Vec[U32(1, 2, 4, 8)]; headers.metadata.num_extra)
            cond(do_ycbcr && !flags.use_lf_frame())
            default(vec![1; headers.metadata.num_extra as usize]),
        pub group_size_shift: ty(u(2)) cond(encoding == Encoding::Modular) default(1),
        pub x_qm_scale:
            ty(u(3))
            cond(!all_default && headers.metadata.xyb_encoded && encoding == Encoding::VarDct)
            default(d_xqms),
        pub b_qm_scale:
            ty(u(3))
            cond(!all_default && headers.metadata.xyb_encoded && encoding == Encoding::VarDct)
            default(2),
        pub passes:
            ty(Bundle(Passes))
            cond(!all_default && frame_type != FrameType::ReferenceOnly),
        pub lf_level: ty(1 + u(2)) cond(frame_type == FrameType::LfFrame) default(0),
        pub have_crop: ty(Bool) cond(!all_default && frame_type != FrameType::LfFrame) default(false),
        pub ux0:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop && frame_type != FrameType::ReferenceOnly),
        pub uy0:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop && frame_type != FrameType::ReferenceOnly),
        pub width:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop),
        pub height:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop),
        pub blending_info:
            ty(Bundle(BlendingInfo))
            cond(!all_default && frame_type.is_normal_frame()),
        pub ec_blending_info:
            ty(Vec[Bundle(BlendingInfo)]; headers.metadata.num_extra)
            cond(!all_default && frame_type.is_normal_frame()),
        pub duration:
            ty(U32(0, 1, u(8), u(32)))
            cond(!all_default && frame_type.is_normal_frame() && headers.metadata.have_animation)
            default(0),
        pub timecode:
            ty(u(32))
            cond(!all_default && frame_type.is_normal_frame() && headers.metadata.animation.have_timecodes)
            default(0),
        pub is_last:
            ty(Bool)
            cond(!all_default && frame_type.is_normal_frame())
            default(frame_type == FrameType::RegularFrame),
        pub save_as_reference:
            ty(u(2))
            cond(!all_default && frame_type != FrameType::LfFrame && !is_last)
            default(0),
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    #[default]
    RegularFrame = 0,
    LfFrame,
    ReferenceOnly,
    SkipProgressive,
}

impl FrameType {
    pub fn is_normal_frame(&self) -> bool {
        match self {
            Self::RegularFrame | Self::SkipProgressive => true,
            _ => false,
        }
    }
}

impl<Ctx> Bundle<Ctx> for FrameType {
    fn parse<R: std::io::Read>(bitstream: &mut crate::Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        Ok(match bitstream.read_bits(2)? {
            0 => Self::RegularFrame,
            1 => Self::LfFrame,
            2 => Self::ReferenceOnly,
            3 => Self::SkipProgressive,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Encoding {
    #[default]
    VarDct = 0,
    Modular,
}

impl<Ctx> Bundle<Ctx> for Encoding {
    fn parse<R: std::io::Read>(bitstream: &mut crate::Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        Ok(match bitstream.read_bits(1)? {
            0 => Self::VarDct,
            1 => Self::Modular,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct FrameFlags(u64);

impl FrameFlags {
    const NOISE: u64 = 0x1;
    const PATCHES: u64 = 0x2;
    const SPLINES: u64 = 0x10;
    const USE_LF_FRAME: u64 = 0x20;
    const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

    pub fn noise(&self) -> bool {
        self.0 & Self::NOISE != 0
    }

    pub fn patches(&self) -> bool {
        self.0 & Self::PATCHES != 0
    }

    pub fn splines(&self) -> bool {
        self.0 & Self::SPLINES != 0
    }

    pub fn use_lf_frame(&self) -> bool {
        self.0 & Self::USE_LF_FRAME != 0
    }

    pub fn skip_adaptive_lf_smoothing(&self) -> bool {
        self.0 & Self::SKIP_ADAPTIVE_LF_SMOOTHING != 0
    }
}

impl<Ctx> Bundle<Ctx> for FrameFlags {
    fn parse<R: std::io::Read>(bitstream: &mut crate::Bitstream<R>, _ctx: &Ctx) -> crate::Result<Self> where Self: Sized {
        crate::read_bits!(bitstream, U64).map(Self)
    }
}
