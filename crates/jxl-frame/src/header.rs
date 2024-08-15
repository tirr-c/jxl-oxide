use crate::Result;
use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle, Name};
use jxl_image::{BitDepth, Extensions, ImageHeader, SizeHeader};

define_bundle! {
    /// Frame header.
    #[derive(Debug)]
    pub struct FrameHeader ctx(headers: &ImageHeader) error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub frame_type: ty(Bundle(FrameType)) cond(!all_default) default(FrameType::RegularFrame),
        pub encoding: ty(Bundle(Encoding)) cond(!all_default) default(Encoding::VarDct),
        pub flags: ty(Bundle(FrameFlags)) cond(!all_default),
        pub do_ycbcr: ty(Bool) cond(!all_default && !headers.metadata.xyb_encoded),
        encoded_color_channels:
            ty(u(0))
            cond(false)
            default({
                let acutally_grayscale = encoding == Encoding::Modular
                    && !do_ycbcr
                    && !headers.metadata.xyb_encoded
                    && headers.metadata.grayscale();
                if acutally_grayscale { 1 } else { 3 }
            }),
        pub jpeg_upsampling: ty(Array[u(2)]; 3) cond(do_ycbcr && !flags.use_lf_frame()),
        pub upsampling: ty(U32(1, 2, 4, 8)) cond(!all_default && !flags.use_lf_frame()) default(1),
        pub ec_upsampling:
            ty(Vec[U32(1, 2, 4, 8)]; headers.metadata.ec_info.len())
            cond(!all_default && !flags.use_lf_frame())
            default(vec![1; headers.metadata.ec_info.len()]),
        pub group_size_shift: ty(u(2)) cond(encoding == Encoding::Modular) default(1),
        pub x_qm_scale:
            ty(u(3))
            cond(!all_default && headers.metadata.xyb_encoded && encoding == Encoding::VarDct)
            default(Self::compute_default_xqms(encoding, headers.metadata.xyb_encoded)),
        pub b_qm_scale:
            ty(u(3))
            cond(!all_default && headers.metadata.xyb_encoded && encoding == Encoding::VarDct)
            default(2),
        pub passes:
            ty(Bundle(Passes))
            cond(!all_default && frame_type != FrameType::ReferenceOnly),
        pub lf_level: ty(1 + u(2)) cond(frame_type == FrameType::LfFrame) default(0),
        pub have_crop: ty(Bool) cond(!all_default && frame_type != FrameType::LfFrame) default(false),
        pub x0:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)); UnpackSigned)
            cond(have_crop && frame_type != FrameType::ReferenceOnly),
        pub y0:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)); UnpackSigned)
            cond(have_crop && frame_type != FrameType::ReferenceOnly),
        pub width:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop)
            default(headers.size.width),
        pub height:
            ty(U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30)))
            cond(have_crop)
            default(headers.size.height),
        pub blending_info:
            ty(Bundle(BlendingInfo))
            ctx((
                !headers.metadata.ec_info.is_empty(),
                None,
                CanvasSizeParams {
                    have_crop,
                    x0,
                    y0,
                    width,
                    height,
                    size: &headers.size,
                },
            ))
            cond(!all_default && frame_type.is_normal_frame()),
        pub ec_blending_info:
            ty(Vec[Bundle(BlendingInfo)]; headers.metadata.ec_info.len())
            ctx((
                !headers.metadata.ec_info.is_empty(),
                Some(blending_info.mode),
                CanvasSizeParams {
                    have_crop,
                    x0,
                    y0,
                    width,
                    height,
                    size: &headers.size,
                },
            ))
            cond(!all_default && frame_type.is_normal_frame()),
        pub duration:
            ty(U32(0, 1, u(8), u(32)))
            cond(!all_default && frame_type.is_normal_frame() && headers.metadata.animation.is_some())
            default(0),
        pub timecode:
            ty(u(32))
            cond(!all_default && frame_type.is_normal_frame() && headers.metadata.animation.as_ref().map(|a| a.have_timecodes).unwrap_or(false))
            default(0),
        pub is_last:
            ty(Bool)
            cond(!all_default && frame_type.is_normal_frame())
            default(frame_type == FrameType::RegularFrame),
        pub save_as_reference:
            ty(u(2))
            cond(!all_default && frame_type != FrameType::LfFrame && !is_last)
            default(0),
        pub resets_canvas:
            ty(Bool)
            cond(false)
            default(Self::resets_canvas(
                blending_info.mode,
                CanvasSizeParams {
                    have_crop,
                    x0,
                    y0,
                    width,
                    height,
                    size: &headers.size,
                },
            )),
        pub save_before_ct:
            ty(Bool)
            cond(
                !all_default && (
                    frame_type == FrameType::ReferenceOnly || (
                        resets_canvas &&
                        (!is_last && (duration == 0 || save_as_reference != 0) && frame_type != FrameType::LfFrame)
                    )
                )
            )
            default(!frame_type.is_normal_frame()),
        pub name: ty(Bundle(Name)) cond(!all_default),
        pub restoration_filter: ty(Bundle(RestorationFilter)) ctx(encoding) cond(!all_default),
        pub extensions: ty(Bundle(Extensions)) cond(!all_default),
        pub bit_depth: ty(Bundle(BitDepth)) cond(false) default(headers.metadata.bit_depth),
    }

    #[derive(Debug)]
    pub struct Passes error(crate::Error) {
        pub num_passes: ty(U32(1, 2, 3, 4 + u(3))) default(1),
        pub num_ds: ty(U32(0, 1, 2, 3 + u(1))) cond(num_passes != 1) default(0),
        pub shift: ty(Vec[u(2)]; num_passes - 1) cond(num_passes != 1) default(vec![0; num_passes as usize - 1]),
        pub downsample: ty(Vec[U32(1, 2, 4, 8)]; num_ds) cond(num_passes != 1) default(vec![1; num_ds as usize]),
        pub last_pass: ty(Vec[U32(0, 1, 2, u(3))]; num_ds) cond(num_passes != 1) default(vec![0; num_ds as usize]),
    }

    #[derive(Debug)]
    pub struct BlendingInfo ctx(context: (bool, Option<BlendMode>, CanvasSizeParams<'_>)) error(crate::Error) {
        pub mode: ty(Bundle(BlendMode)),
        pub alpha_channel:
            ty(U32(0, 1, 2, 3 + u(3)))
            cond(context.0 && (mode == BlendMode::Blend || mode == BlendMode::MulAdd))
            default(0),
        pub clamp:
            ty(Bool)
            cond((context.0 && (mode == BlendMode::Blend || mode == BlendMode::MulAdd)) || mode == BlendMode::Mul)
            default(false),
        pub source:
            ty(u(2))
            cond(!FrameHeader::resets_canvas(context.1.unwrap_or(mode), context.2))
            default(0),
    }

    #[derive(Debug)]
    pub struct RestorationFilter ctx(encoding: Encoding) error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub gab: ty(Bundle(crate::filter::Gabor)) cond(!all_default),
        pub epf: ty(Bundle(crate::filter::EdgePreservingFilter)) cond(!all_default),
        pub extensions: ty(Bundle(Extensions)) cond(!all_default),
    }
}

#[derive(Copy, Clone)]
struct CanvasSizeParams<'a> {
    have_crop: bool,
    x0: i32,
    y0: i32,
    width: u32,
    height: u32,
    size: &'a SizeHeader,
}

impl FrameHeader {
    fn test_full_image(canvas_size: CanvasSizeParams) -> bool {
        let CanvasSizeParams {
            x0,
            y0,
            width,
            height,
            size,
            ..
        } = canvas_size;

        if x0 > 0 || y0 > 0 {
            return false;
        }

        let right = x0 as i64 + (width as i64);
        let bottom = y0 as i64 + (height as i64);
        (right >= size.width as i64) && (bottom >= size.height as i64)
    }

    fn resets_canvas(blending_mode: BlendMode, canvas_size: CanvasSizeParams) -> bool {
        blending_mode == BlendMode::Replace
            && (!canvas_size.have_crop || Self::test_full_image(canvas_size))
    }

    fn compute_default_xqms(encoding: Encoding, xyb_encoded: bool) -> u32 {
        if xyb_encoded && encoding == Encoding::VarDct {
            3
        } else {
            2
        }
    }

    /// Returns whether this frame is a keyframe that should be displayed.
    #[inline]
    pub fn is_keyframe(&self) -> bool {
        self.frame_type.is_normal_frame() && (self.is_last || self.duration != 0)
    }

    #[inline]
    pub fn can_reference(&self) -> bool {
        !self.is_last
            && (self.duration == 0 || self.save_as_reference != 0)
            && self.frame_type != FrameType::LfFrame
    }

    pub fn sample_width(&self, upsampling: u32) -> u32 {
        let &Self {
            mut width,
            lf_level,
            ..
        } = self;

        if upsampling > 1 {
            width = (width + upsampling - 1) / upsampling;
        }
        if lf_level > 0 {
            let div = 1u32 << (3 * lf_level);
            width = (width + div - 1) / div;
        }

        width
    }

    pub fn sample_height(&self, upsampling: u32) -> u32 {
        let &Self {
            mut height,
            lf_level,
            ..
        } = self;

        if upsampling > 1 {
            height = (height + upsampling - 1) / upsampling;
        }
        if lf_level > 0 {
            let div = 1u32 << (3 * lf_level);
            height = (height + div - 1) / div;
        }

        height
    }

    pub fn color_sample_width(&self) -> u32 {
        self.sample_width(self.upsampling)
    }

    pub fn color_sample_height(&self) -> u32 {
        self.sample_height(self.upsampling)
    }

    /// Returns the number of channels actually encoded in the frame.
    #[inline]
    pub fn encoded_color_channels(&self) -> usize {
        self.encoded_color_channels as usize
    }

    pub fn num_groups(&self) -> u32 {
        let width = self.color_sample_width();
        let height = self.color_sample_height();
        let group_dim = self.group_dim();

        let hgroups = (width + group_dim - 1) / group_dim;
        let vgroups = (height + group_dim - 1) / group_dim;

        hgroups * vgroups
    }

    pub fn num_lf_groups(&self) -> u32 {
        let width = self.color_sample_width();
        let height = self.color_sample_height();
        let lf_group_dim = self.lf_group_dim();

        let hgroups = (width + lf_group_dim - 1) / lf_group_dim;
        let vgroups = (height + lf_group_dim - 1) / lf_group_dim;

        hgroups * vgroups
    }

    pub fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }

    pub fn groups_per_row(&self) -> u32 {
        let group_dim = self.group_dim();
        (self.color_sample_width() + group_dim - 1) / group_dim
    }

    pub fn lf_group_dim(&self) -> u32 {
        self.group_dim() * 8
    }

    pub fn lf_groups_per_row(&self) -> u32 {
        let lf_group_dim = self.lf_group_dim();
        (self.color_sample_width() + lf_group_dim - 1) / lf_group_dim
    }

    pub fn group_size_for(&self, group_idx: u32) -> (u32, u32) {
        self.size_for(self.group_dim(), group_idx)
    }

    pub fn lf_group_size_for(&self, lf_group_idx: u32) -> (u32, u32) {
        self.size_for(self.lf_group_dim(), lf_group_idx)
    }

    fn size_for(&self, group_dim: u32, group_idx: u32) -> (u32, u32) {
        let width = self.color_sample_width();
        let height = self.color_sample_height();
        let full_rows = height / group_dim;
        let rows_remainder = height % group_dim;
        let full_cols = width / group_dim;
        let cols_remainder = width % group_dim;

        let stride = full_cols + (cols_remainder > 0) as u32;
        let row = group_idx / stride;
        let col = group_idx % stride;

        let group_width = if col >= full_cols {
            cols_remainder
        } else {
            group_dim
        };
        let group_height = if row >= full_rows {
            rows_remainder
        } else {
            group_dim
        };
        (group_width, group_height)
    }

    pub fn lf_group_idx_from_group_idx(&self, group_idx: u32) -> u32 {
        let groups_per_row = self.groups_per_row();
        let lf_group_col = (group_idx % groups_per_row) / 8;
        let lf_group_row = (group_idx / groups_per_row) / 8;
        lf_group_col + lf_group_row * self.lf_groups_per_row()
    }

    pub fn is_group_collides_region(&self, group_idx: u32, region: (u32, u32, u32, u32)) -> bool {
        let group_dim = self.group_dim();
        let group_per_row = self.groups_per_row();
        let group_left = (group_idx % group_per_row) * group_dim;
        let group_top = (group_idx / group_per_row) * group_dim;
        is_aabb_collides(region, (group_left, group_top, group_dim, group_dim))
    }

    pub fn is_lf_group_collides_region(
        &self,
        lf_group_idx: u32,
        region: (u32, u32, u32, u32),
    ) -> bool {
        let lf_group_dim = self.lf_group_dim();
        let lf_group_per_row = self.lf_groups_per_row();
        let group_left = (lf_group_idx % lf_group_per_row) * lf_group_dim;
        let group_top = (lf_group_idx / lf_group_per_row) * lf_group_dim;
        is_aabb_collides(region, (group_left, group_top, lf_group_dim, lf_group_dim))
    }
}

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
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
        matches!(self, Self::RegularFrame | Self::SkipProgressive)
    }

    pub fn is_progressive_frame(&self) -> bool {
        matches!(self, Self::RegularFrame | Self::LfFrame)
    }
}

impl<Ctx> Bundle<Ctx> for FrameType {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, _ctx: Ctx) -> Result<Self> {
        Ok(match bitstream.read_bits(2)? {
            0 => Self::RegularFrame,
            1 => Self::LfFrame,
            2 => Self::ReferenceOnly,
            3 => Self::SkipProgressive,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
#[repr(u8)]
pub enum Encoding {
    #[default]
    VarDct = 0,
    Modular,
}

impl<Ctx> Bundle<Ctx> for Encoding {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, _ctx: Ctx) -> Result<Self> {
        Ok(match bitstream.read_bits(1)? {
            0 => Self::VarDct,
            1 => Self::Modular,
            _ => unreachable!(),
        })
    }
}

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
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
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, _ctx: Ctx) -> Result<Self> {
        Ok(Self(bitstream.read_u64()?))
    }
}

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
#[repr(u8)]
pub enum BlendMode {
    #[default]
    Replace = 0,
    Add = 1,
    Blend = 2,
    MulAdd = 3,
    Mul = 4,
}

impl<Ctx> Bundle<Ctx> for BlendMode {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, _ctx: Ctx) -> Result<Self> {
        Ok(match read_bits!(bitstream, U32(0, 1, 2, 3 + u(2)))? {
            0 => Self::Replace,
            1 => Self::Add,
            2 => Self::Blend,
            3 => Self::MulAdd,
            4 => Self::Mul,
            value => {
                return Err(jxl_bitstream::Error::InvalidEnum {
                    name: "BlendMode",
                    value,
                }
                .into())
            }
        })
    }
}

impl BlendMode {
    #[inline]
    pub fn use_alpha(self) -> bool {
        matches!(self, Self::Blend | Self::MulAdd)
    }
}

fn is_aabb_collides(rect0: (u32, u32, u32, u32), rect1: (u32, u32, u32, u32)) -> bool {
    let (x0, y0, w0, h0) = rect0;
    let (x1, y1, w1, h1) = rect1;
    (x0 < x1 + w1) && (x0 + w0 > x1) && (y0 < y1 + h1) && (y0 + h0 > y1)
}
