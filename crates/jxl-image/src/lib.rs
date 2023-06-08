//! This crate provides types related to JPEG XL image headers, such as
//! [image size information][SizeHeader], [color encoding][ColourEncoding] and
//! [animation TPS (ticks per second) information][AnimationHeader]. Most of the information is in
//! the [`ImageMetadata`] struct.
//!
//! Image header is at the beginning of the bitstream. One can parse [`ImageHeader`] from the
//! bitstream to retrieve information about the image.
use std::io::Read;
use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle, Result, Name};
use jxl_color::header::*;

/// JPEG XL image header.
///
/// Use [`Bundle::parse`] to parse the header.
#[derive(Debug)]
pub struct ImageHeader {
    /// Image size information.
    pub size: SizeHeader,
    /// Image metadata.
    pub metadata: ImageMetadata,
}

impl<Ctx> Bundle<Ctx> for ImageHeader {
    type Error = jxl_bitstream::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> Result<Self> {
        let signature = bitstream.read_bits(16)?;
        if signature != 0xaff {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "JPEG XL signature mismatch"
            ));
        }

        let size = SizeHeader::parse(bitstream, ())?;
        let metadata = ImageMetadata::parse(bitstream, ())?;

        let tone_mapping = &metadata.tone_mapping;
        if tone_mapping.intensity_target <= 0.0 {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid intensity target"
            ));
        }
        if tone_mapping.min_nits < 0.0
            || tone_mapping.min_nits > tone_mapping.intensity_target
        {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid tone mapping min_nits"
            ));
        }
        if tone_mapping.linear_below < 0.0
            || (tone_mapping.relative_to_max_display && tone_mapping.linear_below > 1.0)
        {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid tone mapping linear_below"
            ));
        }

        Ok(Self { size, metadata })
    }
}

define_bundle! {
    /// Image size information.
    #[derive(Debug)]
    pub struct SizeHeader {
        div8: ty(Bool) default(false),
        h_div8: ty(1 + u(5)) cond(div8) default(0),
        /// Image height.
        pub height:
            ty(U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))) cond(!div8)
            default(8 * h_div8),
        ratio: ty(u(3)) default(0),
        w_div8: ty(1 + u(5)) cond(div8 && ratio == 0) default(0),
        /// Image width.
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
    /// Image metadata.
    #[derive(Debug)]
    pub struct ImageMetadata {
        all_default: ty(Bool) default(true),
        extra_fields: ty(Bool) cond(!all_default) default(false),
        /// Value representing image orientation.
        pub orientation: ty(1 + u(3)) cond(extra_fields) default(1),
        have_intr_size: ty(Bool) cond(extra_fields) default(false),
        /// Recommended size to display the image.
        pub intrinsic_size: ty(Bundle(Option<SizeHeader>)) cond(have_intr_size),
        have_preview: ty(Bool) cond(extra_fields) default(false),
        /// Size information of the preview frame, if there is any.
        pub preview: ty(Bundle(Option<PreviewHeader>)) cond(have_preview),
        have_animation: ty(Bool) cond(extra_fields) default(false),
        /// Information about the animation such as TPS, if the image is animated.
        pub animation: ty(Bundle(Option<AnimationHeader>)) cond(have_animation),
        /// Bit depth information, which is used to parse Modular image samples.
        pub bit_depth: ty(Bundle(BitDepth)) cond(!all_default),
        /// Whether 16-bit buffer is sufficient to correctly parse Modular images.
        pub modular_16bit_buffers: ty(Bool) cond(!all_default) default(true),
        num_extra: ty(U32(0, 1, 2 + u(4), 1 + u(12))) cond(!all_default) default(0),
        /// Information about extra channels, such as alpha and black channels.
        pub ec_info: ty(Vec[Bundle(ExtraChannelInfo)]; num_extra) cond(!all_default),
        /// Whether the image is encoded in XYB color space.
        pub xyb_encoded: ty(Bool) cond(!all_default) default(true),
        /// Color encoding of the image.
        ///
        /// If `xyb_encoded` is `true`, this is a suggestion of the color space to present the
        /// decoded image. If it's not, the decoded image is in the color space represented by this
        /// field.
        pub colour_encoding: ty(Bundle(ColourEncoding)) cond(!all_default),
        /// Tone mapping information, which is used to map HDR images to SDR.
        pub tone_mapping: ty(Bundle(ToneMapping)) cond(extra_fields),
        pub extensions: ty(Bundle(Extensions)) cond(!all_default),
        default_m: ty(Bool),
        /// Opsin inverse matrix, which is used to transform XYB encoded image to sRGB color space.
        pub opsin_inverse_matrix: ty(Bundle(OpsinInverseMatrix)) cond(!default_m && xyb_encoded),
        cw_mask: ty(u(3)) cond(!default_m) default(0),
        /// 2x upsampling weights.
        pub up2_weight: ty(Array[F16]; 15) cond(cw_mask & 1 != 0) default(Self::D_UP2),
        /// 4x upsampling weights.
        pub up4_weight: ty(Array[F16]; 55) cond(cw_mask & 2 != 0) default(Self::D_UP4),
        /// 8x upsampling weights.
        pub up8_weight: ty(Array[F16]; 210) cond(cw_mask & 4 != 0) default(Self::D_UP8),
    }

    #[derive(Debug)]
    pub struct PreviewHeader {
        div8: ty(Bool),
        h_div8: ty(U32(16, 32, 1 + u(5), 33 + u(9))) cond(div8) default(1),
        /// Height of the preview image.
        pub height:
            ty(U32(1 + u(6), 65 + u(8), 321 + u(10), 1345 + u(12))) cond(!div8)
            default(8 * h_div8),
        ratio: ty(u(3)),
        w_div8: ty(U32(16, 32, 1 + u(5), 33 + u(9))) cond(div8) default(1),
        /// Width of the preview image.
        pub width:
            ty(U32(1 + u(6), 65 + u(8), 321 + u(10), 1345 + u(12))) cond(!div8)
            default(SizeHeader::compute_default_width(ratio, w_div8, height)),
    }

    /// Animation information.
    ///
    /// TPS (ticks per second) is computed as `tps_numerator / tps_denominator`, which means
    /// `tps_denominator / tps_numerator` seconds per tick.
    #[derive(Debug)]
    pub struct AnimationHeader {
        /// TPS numerator.
        pub tps_numerator: ty(U32(100, 1000, 1 + u(10), 1 + u(30))) default(0),
        /// TPS denominator.
        pub tps_denominator: ty(U32(1, 1001, 1 + u(8), 1 + u(10))) default(0),
        /// Number of loops, where 0 means it loops forever.
        pub num_loops: ty(U32(0, u(3), u(16), u(32))) default(0),
        /// Whether keyframes in the image has their timecodes embedded.
        pub have_timecodes: ty(Bool) default(false),
    }

    #[derive(Debug)]
    pub struct Extensions {
        extensions: ty(U64) default(0),
        extension_bits:
            ty(Vec[U64]; (extensions + 7) / 8) cond(extensions != 0)
            default(vec![0; ((extensions + 7) / 8) as usize]),
    }
}

impl ImageMetadata {
    /// Returns whether the image is grayscale.
    #[inline]
    pub fn grayscale(&self) -> bool {
        self.colour_encoding.colour_space == ColourSpace::Grey
    }

    /// Returns the number of channels actually encoded in the image.
    #[inline]
    pub fn encoded_color_channels(&self) -> usize {
        if !self.xyb_encoded && self.grayscale() {
            1
        } else {
            3
        }
    }

    /// Returns the index of the first alpha channel in the image.
    pub fn alpha(&self) -> Option<usize> {
        self.ec_info.iter()
            .position(|info| matches!(info.ty, ExtraChannelType::Alpha { .. }))
    }

    /// Returns where the given coordinate will be placed after the orientation is applied.
    #[inline]
    pub fn apply_orientation(&self, width: u32, height: u32, left: u32, top: u32, inverse: bool) -> (u32, u32, u32, u32) {
        let (left, top) = match self.orientation {
            1 => (left, top),
            2 => (width - left - 1, top),
            3 => (width - left - 1, height - top - 1),
            4 => (left, height - top - 1),
            5 => (top, left),
            6 if inverse => (top, width - left - 1),
            6 => (height - top - 1, left),
            7 => (height - top - 1, width - left - 1),
            8 if inverse => (height - top - 1, left),
            8 => (top, width - left - 1),
            _ => unreachable!(),
        };
        let (width, height) = match self.orientation {
            1..=4 => (width, height),
            5..=8 => (height, width),
            _ => unreachable!(),
        };
        (width, height, left, top)
    }
}

/// Information about an extra channel.
#[derive(Debug, Default, Clone)]
pub struct ExtraChannelInfo {
    /// Type and associated parameters of the channel.
    pub ty: ExtraChannelType,
    /// Bit depth information about the channel.
    pub bit_depth: BitDepth,
    /// `dim_shift` used to decode Modular image.
    pub dim_shift: u32,
    /// Name of the channel.
    pub name: Name,
}

impl<Ctx> Bundle<Ctx> for ExtraChannelInfo {
    type Error = jxl_bitstream::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> Result<Self> {
        let default_alpha_channel = bitstream.read_bool()?;
        if default_alpha_channel {
            return Ok(Self::default());
        }

        let ty_id = read_bits!(bitstream, Enum(ExtraChannelTypeRaw))?;
        let bit_depth = BitDepth::parse(bitstream, ())?;
        let dim_shift = read_bits!(bitstream, U32(0, 3, 4, 1 + u(3)))?;
        let name = Name::parse(bitstream, ())?;

        let ty = match ty_id {
            ExtraChannelTypeRaw::Alpha => {
                ExtraChannelType::Alpha {
                    alpha_associated: bitstream.read_bool()?,
                }
            },
            ExtraChannelTypeRaw::Depth => ExtraChannelType::Depth,
            ExtraChannelTypeRaw::SpotColour => {
                ExtraChannelType::SpotColour {
                    red: bitstream.read_f16_as_f32()?,
                    green: bitstream.read_f16_as_f32()?,
                    blue: bitstream.read_f16_as_f32()?,
                    solidity: bitstream.read_f16_as_f32()?,
                }
            },
            ExtraChannelTypeRaw::SelectionMask => ExtraChannelType::SelectionMask,
            ExtraChannelTypeRaw::Black => ExtraChannelType::Black,
            ExtraChannelTypeRaw::Cfa => {
                ExtraChannelType::Cfa {
                    cfa_channel: read_bits!(bitstream, U32(1, u(2), 3 + u(4), 19 + u(8)))?,
                }
            },
            ExtraChannelTypeRaw::Thermal => ExtraChannelType::Thermal,
            ExtraChannelTypeRaw::NonOptional => ExtraChannelType::NonOptional,
            ExtraChannelTypeRaw::Optional => ExtraChannelType::Optional,
        };

        Ok(Self {
            ty,
            bit_depth,
            dim_shift,
            name,
        })
    }
}

impl ExtraChannelInfo {
    /// Returns whether this is an alpha channel.
    #[inline]
    pub fn is_alpha(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Alpha { .. })
    }

    /// Returns whether the alpha channel has premultiplied semantics.
    #[inline]
    pub fn alpha_associated(&self) -> Option<bool> {
        if let ExtraChannelType::Alpha { alpha_associated } = self.ty {
            Some(alpha_associated)
        } else {
            None
        }
    }

    /// Returns whether this is a black channel of a CMYK image.
    #[inline]
    pub fn is_black(&self) -> bool {
        self.ty == ExtraChannelType::Black
    }
}

/// Type of an extra channel.
#[derive(Debug, PartialEq, Copy, Clone)]
#[repr(u8)]
pub enum ExtraChannelType {
    Alpha {
        alpha_associated: bool,
    } = 0,
    Depth,
    SpotColour {
        red: f32,
        green: f32,
        blue: f32,
        solidity: f32,
    },
    SelectionMask,
    Black,
    Cfa {
        cfa_channel: u32,
    },
    Thermal,
    NonOptional = 15,
    Optional,
}

impl Default for ExtraChannelType {
    fn default() -> Self {
        Self::Alpha { alpha_associated: false }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
enum ExtraChannelTypeRaw {
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

impl TryFrom<u32> for ExtraChannelTypeRaw {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
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

/// Bit depth information.
#[derive(Debug, Copy, Clone)]
pub enum BitDepth {
    /// Modular image samples represent integer values, where the range
    /// `0..=(1 << bits_per_sample) - 1` corresponds to \[0.0, 1.0\], scaled linearly.
    ///
    /// The value outside of the \[0.0, 1.0\] range is *not* clamped.
    IntegerSample {
        bits_per_sample: u32,
    },
    /// Modular image samples represent bitcast of floating point values with a sign bit,
    /// `exp_bits` exponential bits, and the remaining mantissa bits.
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

impl BitDepth {
    #[inline]
    pub fn bits_per_sample(self) -> u32 {
        match self {
            Self::IntegerSample { bits_per_sample } => bits_per_sample,
            Self::FloatSample { bits_per_sample, .. } => bits_per_sample,
        }
    }

    /// Parses the given Modular image sample to an `f32`.
    #[inline]
    pub fn parse_integer_sample(self, sample: i32) -> f32 {
        match self {
            Self::IntegerSample { bits_per_sample } => {
                let div = (1i32 << bits_per_sample) - 1;
                (sample as f64 / div as f64) as f32
            },
            Self::FloatSample { bits_per_sample, exp_bits } => {
                let sample = sample as u32;
                let mantissa_bits = bits_per_sample - exp_bits - 1;
                let mantissa_mask = (1u32 << mantissa_bits) - 1;
                let exp_mask = ((1u32 << (bits_per_sample - 1)) - 1) ^ mantissa_mask;

                let is_signed = (sample & (1u32 << (bits_per_sample - 1))) != 0;
                let mantissa = sample & mantissa_mask;
                let exp = ((sample & exp_mask) >> mantissa_bits) as i32;
                let exp = exp - ((1 << (exp_bits - 1)) - 1);

                // TODO: handle subnormal values.
                let f32_mantissa_bits = f32::MANTISSA_DIGITS - 1;
                let mantissa = match mantissa_bits.cmp(&f32_mantissa_bits) {
                    std::cmp::Ordering::Less => mantissa << (f32_mantissa_bits - mantissa_bits),
                    std::cmp::Ordering::Greater => mantissa >> (mantissa_bits - f32_mantissa_bits),
                    _ => mantissa,
                };
                let exp = (exp + 127) as u32;
                let sign = is_signed as u32;

                let bits = (sign << 31) | (exp << f32_mantissa_bits) | mantissa;
                f32::from_bits(bits)
            },
        }
    }
}

impl<Ctx> Bundle<Ctx> for BitDepth {
    type Error = jxl_bitstream::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _ctx: Ctx) -> Result<Self> {
        if bitstream.read_bool()? { // float_sample
            let bits_per_sample = read_bits!(bitstream, U32(32, 16, 24, 1 + u(6)))?;
            let exp_bits = read_bits!(bitstream, 1 + u(4))?;
            Ok(Self::FloatSample { bits_per_sample, exp_bits })
        } else {
            let bits_per_sample = read_bits!(bitstream, U32(8, 10, 12, 1 + u(6)))?;
            Ok(Self::IntegerSample { bits_per_sample })
        }
    }
}

#[allow(clippy::excessive_precision)]
impl ImageMetadata {
    const D_UP2: [f32; 15] = [
        -0.01716200, -0.03452303, -0.04022174, -0.02921014, -0.00624645,
        0.14111091, 0.28896755, 0.00278718, -0.01610267, 0.56661550,
        0.03777607, -0.01986694, -0.03144731, -0.01185068, -0.00213539,
    ];
    const D_UP4: [f32; 55] = [
        -0.02419067, -0.03491987, -0.03693351, -0.03094285, -0.00529785,
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
