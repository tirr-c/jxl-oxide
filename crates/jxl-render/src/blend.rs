use std::sync::Arc;

use jxl_frame::{
    Frame,
    data::{BlendingModeInformation, PatchRef},
    header::{BlendMode as FrameBlendMode, BlendingInfo},
};
use jxl_grid::{AlignedGrid, MutableSubgrid, SharedSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, Reference, Region, Result, image::ImageBuffer};

#[derive(Debug)]
enum BlendMode<'a> {
    Replace,
    Add,
    Mul(bool),
    Blend(BlendAlpha<'a>),
    MulAdd(BlendAlpha<'a>),
    MixAlpha { clamp: bool, swapped: bool },
    Skip,
}

#[derive(Debug)]
struct BlendParams<'a> {
    mode: BlendMode<'a>,
    base_topleft: (usize, usize),
    new_topleft: (usize, usize),
    width: usize,
    height: usize,
}

struct BlendAlpha<'a> {
    base: Option<SharedSubgrid<'a, f32>>,
    new: Option<SharedSubgrid<'a, f32>>,
    clamp: bool,
    swapped: bool,
    premultiplied: bool,
}

impl std::fmt::Debug for BlendAlpha<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlendAlpha")
            .field("clamp", &self.clamp)
            .field("swapped", &self.swapped)
            .field("premultiplied", &self.premultiplied)
            .finish_non_exhaustive()
    }
}

impl<'a> BlendParams<'a> {
    fn from_blending_info(
        channel_idx: usize,
        color_channels: usize,
        blending_info: &BlendingInfo,
        base_alpha: Option<SharedSubgrid<'a, f32>>,
        new_alpha: Option<SharedSubgrid<'a, f32>>,
        premultiplied: Option<bool>,
    ) -> Self {
        let mode = match blending_info.mode {
            FrameBlendMode::Replace => BlendMode::Replace,
            FrameBlendMode::Add => BlendMode::Add,
            FrameBlendMode::Blend
                if channel_idx == blending_info.alpha_channel as usize + color_channels =>
            {
                BlendMode::MixAlpha {
                    clamp: blending_info.clamp,
                    swapped: false,
                }
            }
            FrameBlendMode::Blend => BlendMode::Blend(BlendAlpha {
                base: base_alpha,
                new: new_alpha,
                clamp: blending_info.clamp,
                swapped: false,
                premultiplied: premultiplied.unwrap_or(false),
            }),
            FrameBlendMode::MulAdd
                if channel_idx == blending_info.alpha_channel as usize + color_channels =>
            {
                BlendMode::Skip
            }
            FrameBlendMode::MulAdd => BlendMode::MulAdd(BlendAlpha {
                base: base_alpha,
                new: new_alpha,
                clamp: blending_info.clamp,
                swapped: false,
                premultiplied: premultiplied.unwrap_or(false),
            }),
            FrameBlendMode::Mul => BlendMode::Mul(blending_info.clamp),
        };

        Self {
            mode,
            base_topleft: (0, 0),
            new_topleft: (0, 0),
            width: 0,
            height: 0,
        }
    }

    fn from_patch_blending_info(
        channel_idx: usize,
        color_channels: usize,
        blending_info: &BlendingModeInformation,
        base_alpha: Option<SharedSubgrid<'a, f32>>,
        new_alpha: Option<SharedSubgrid<'a, f32>>,
        premultiplied: Option<bool>,
    ) -> Option<Self> {
        use jxl_frame::data::PatchBlendMode;

        let mode = match blending_info.mode {
            PatchBlendMode::None => return None,
            PatchBlendMode::Replace => BlendMode::Replace,
            PatchBlendMode::Add => BlendMode::Add,
            PatchBlendMode::Mul => BlendMode::Mul(blending_info.clamp),
            PatchBlendMode::BlendAbove | PatchBlendMode::BlendBelow => {
                let swapped = blending_info.mode == PatchBlendMode::BlendBelow;
                if channel_idx == blending_info.alpha_channel as usize + color_channels {
                    BlendMode::MixAlpha {
                        clamp: blending_info.clamp,
                        swapped,
                    }
                } else {
                    BlendMode::Blend(BlendAlpha {
                        base: base_alpha,
                        new: new_alpha,
                        clamp: blending_info.clamp,
                        swapped,
                        premultiplied: premultiplied.unwrap_or(false),
                    })
                }
            }
            PatchBlendMode::MulAddAbove | PatchBlendMode::MulAddBelow => {
                let swapped = blending_info.mode == PatchBlendMode::MulAddBelow;
                if channel_idx == blending_info.alpha_channel as usize + color_channels {
                    if swapped {
                        BlendMode::Replace
                    } else {
                        BlendMode::Skip
                    }
                } else {
                    BlendMode::MulAdd(BlendAlpha {
                        base: base_alpha,
                        new: new_alpha,
                        clamp: blending_info.clamp,
                        swapped,
                        premultiplied: premultiplied.unwrap_or(false),
                    })
                }
            }
        };

        Some(Self {
            mode,
            base_topleft: (0, 0),
            new_topleft: (0, 0),
            width: 0,
            height: 0,
        })
    }
}

fn source_and_alpha_from_blending_info(
    blending_info: &BlendingInfo,
    has_extra: bool,
) -> (usize, Option<usize>) {
    use jxl_frame::header::BlendMode;

    let source = blending_info.source as usize;
    let alpha = (matches!(blending_info.mode, BlendMode::Blend | BlendMode::MulAdd) && has_extra)
        .then_some(blending_info.alpha_channel as usize);

    (source, alpha)
}

pub(crate) fn blend<S: Sample>(
    image_header: &ImageHeader,
    reference_grids: [Option<Reference<S>>; 4],
    new_frame: &Frame,
    new_grid: &mut ImageWithRegion,
    output_frame_region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let header = new_frame.header();
    let tracker = new_frame.alloc_tracker();

    let full_frame_region = Region::with_size(header.width, header.height);
    let output_image_region = output_frame_region.translate(header.x0, header.y0);

    let mut used_refs = [false; 4];
    for blending_info in std::iter::once(&header.blending_info).chain(&header.ec_blending_info) {
        let ref_idx = blending_info.source as usize;
        used_refs[ref_idx] = true;
    }

    let mut ref_list = Vec::new();
    for ref_idx in 0..4 {
        if !used_refs[ref_idx] {
            continue;
        }
        let ref_grid = &reference_grids[ref_idx];
        if let Some(grid) = ref_grid {
            ref_list.push(grid);
        }
    }
    ref_list.sort_by_key(|grid| grid.frame.idx);

    for grid in ref_list {
        Arc::clone(&grid.image)
            .run_with_image()?
            .blend(Some(output_image_region), pool)?;
    }

    let color_channels = new_grid.color_channels();
    let mut output_grid = ImageWithRegion::new(color_channels, tracker);
    output_grid.set_ct_done(new_grid.ct_done());

    let has_extra = !header.ec_blending_info.is_empty();
    for (idx, blending_info) in std::iter::repeat_n(&header.blending_info, color_channels)
        .chain(&header.ec_blending_info)
        .enumerate()
    {
        let bit_depth = if let Some(ec_idx) = idx.checked_sub(color_channels) {
            image_header.metadata.ec_info[ec_idx].bit_depth
        } else {
            image_header.metadata.bit_depth
        };

        let (ref_idx, alpha_idx) = source_and_alpha_from_blending_info(blending_info, has_extra);
        let ref_grid = &reference_grids[ref_idx];
        let mut can_overwrite = idx < color_channels
            && (header.is_last
                || (header.can_reference() && ref_idx == header.save_as_reference as usize));

        let original_frame_region = new_grid.regions_and_shifts()[idx]
            .0
            .intersection(full_frame_region);
        let clipped_original_frame_region = original_frame_region.intersection(output_frame_region);

        let mut base_alpha_grid;
        let mut base_alpha = None;
        let mut target_grid;
        let target_region;
        let target_subgrid;
        let mut clone_empty = false;
        if let Some(grid) = ref_grid {
            if grid.frame.header().is_keyframe() {
                can_overwrite = false;
            }

            let base_grid_render = Arc::clone(&grid.image).run_with_image()?;
            let base_grid = base_grid_render.blend(Some(output_image_region), pool)?;
            assert_eq!(base_grid.color_channels(), color_channels);

            if base_grid.regions_and_shifts()[idx].0.is_empty() {
                clone_empty = true;
                target_grid = ImageBuffer::F32(AlignedGrid::with_alloc_tracker(
                    output_frame_region.width as usize,
                    output_frame_region.height as usize,
                    tracker,
                )?);
                target_region = output_frame_region;
                target_subgrid = target_grid.as_float_mut().unwrap().as_subgrid_mut();
            } else {
                let base_frame_header = grid.frame.header();
                let base_frame_region =
                    output_image_region.translate(-base_frame_header.x0, -base_frame_header.y0);

                target_grid = if can_overwrite {
                    if let Some(mut image) = base_grid_render.try_take_blended() {
                        let buffer = &mut image.buffer_mut()[idx];
                        std::mem::replace(buffer, ImageBuffer::F32(AlignedGrid::empty()))
                    } else {
                        base_grid.buffer()[idx].try_clone()?
                    }
                } else {
                    base_grid.buffer()[idx].try_clone()?
                };
                target_region = base_grid.regions_and_shifts()[idx].0.translate(
                    base_frame_header.x0 - header.x0,
                    base_frame_header.y0 - header.y0,
                );
                let target_grid = target_grid.convert_to_float_modular(bit_depth)?;
                target_subgrid = {
                    let grid_region = base_grid.regions_and_shifts()[idx].0;
                    let region = base_frame_region.translate(-grid_region.left, -grid_region.top);
                    let Region {
                        left,
                        top,
                        width,
                        height,
                    } = region;
                    let right = left.wrapping_add_unsigned(width);
                    let bottom = top.wrapping_add_unsigned(height);
                    target_grid
                        .as_subgrid_mut()
                        .subgrid(left as usize..right as usize, top as usize..bottom as usize)
                };

                if let Some(alpha_idx) = alpha_idx {
                    if alpha_idx + color_channels != idx {
                        let alpha_bit_depth = image_header.metadata.ec_info[alpha_idx].bit_depth;
                        base_alpha_grid =
                            base_grid.buffer()[alpha_idx + color_channels].try_clone()?;
                        let base_alpha_grid =
                            base_alpha_grid.convert_to_float_modular(alpha_bit_depth)?;
                        base_alpha = Some({
                            let grid_region =
                                base_grid.regions_and_shifts()[alpha_idx + color_channels].0;
                            let region =
                                base_frame_region.translate(-grid_region.left, -grid_region.top);
                            let Region {
                                left,
                                top,
                                width,
                                height,
                            } = region;
                            let right = left.wrapping_add_unsigned(width);
                            let bottom = top.wrapping_add_unsigned(height);
                            base_alpha_grid.as_subgrid().subgrid(
                                left as usize..right as usize,
                                top as usize..bottom as usize,
                            )
                        });
                    }
                }
            }
        } else {
            clone_empty = true;
            target_grid = ImageBuffer::F32(AlignedGrid::with_alloc_tracker(
                output_frame_region.width as usize,
                output_frame_region.height as usize,
                tracker,
            )?);
            target_region = output_frame_region;
            target_subgrid = target_grid.as_float_mut().unwrap().as_subgrid_mut();
        }

        if let Some(idx) = alpha_idx {
            let bit_depth = image_header.metadata.ec_info[idx].bit_depth;
            new_grid.buffer_mut()[idx + color_channels].convert_to_float_modular(bit_depth)?;
        }
        new_grid.buffer_mut()[idx].convert_to_float_modular(bit_depth)?;

        let mut blend_params = if clone_empty {
            let new_alpha = alpha_idx.map(|idx| {
                new_grid.buffer()[idx + color_channels]
                    .as_float()
                    .unwrap()
                    .as_subgrid()
            });
            let premultiplied =
                alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(
                idx,
                color_channels,
                blending_info,
                None,
                new_alpha,
                premultiplied,
            )
        } else {
            let new_alpha = alpha_idx.map(|idx| {
                new_grid.buffer()[idx + color_channels]
                    .as_float()
                    .unwrap()
                    .as_subgrid()
            });
            let premultiplied =
                alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(
                idx,
                color_channels,
                blending_info,
                base_alpha,
                new_alpha,
                premultiplied,
            )
        };
        blend_params.base_topleft = (
            clipped_original_frame_region
                .left
                .abs_diff(output_frame_region.left) as usize,
            clipped_original_frame_region
                .top
                .abs_diff(output_frame_region.top) as usize,
        );
        blend_params.new_topleft = (
            clipped_original_frame_region
                .left
                .abs_diff(original_frame_region.left) as usize,
            clipped_original_frame_region
                .top
                .abs_diff(original_frame_region.top) as usize,
        );
        blend_params.width = clipped_original_frame_region.width as usize;
        blend_params.height = clipped_original_frame_region.height as usize;

        let new_grid = new_grid.buffer()[idx].as_float().unwrap();
        blend_single(target_subgrid, new_grid.as_subgrid(), &blend_params);
        output_grid.append_channel(target_grid, target_region);
    }

    if header.can_reference() {
        let ref_idx = header.save_as_reference as usize;
        if let Some(grid) = &reference_grids[ref_idx] {
            if !grid.frame.header().is_keyframe() {
                grid.image.reset();
            }
        }
    }

    output_grid.set_blend_done(true);
    Ok(output_grid)
}

pub fn patch(
    image_header: &ImageHeader,
    base_grid: &mut ImageWithRegion,
    patch_ref_grid: &ImageWithRegion,
    patch_ref: &PatchRef,
) -> Result<()> {
    use jxl_frame::data::PatchBlendMode;

    let color_channels = base_grid.color_channels();
    assert_eq!(patch_ref_grid.color_channels(), color_channels);
    for target in &patch_ref.patch_targets {
        for (idx, blending_info) in std::iter::repeat_n(&target.blending[0], color_channels)
            .chain(&target.blending[1..])
            .enumerate()
        {
            let base_grid_region = base_grid.regions_and_shifts()[idx].0;
            let ref_grid_region = patch_ref_grid.regions_and_shifts()[idx].0;

            let target_patch_region = base_grid_region.intersection(Region {
                left: target.x,
                top: target.y,
                width: patch_ref.width,
                height: patch_ref.height,
            });
            let width = target_patch_region.width;
            let height = target_patch_region.height;

            let left = target_patch_region.left - target.x;
            let top = target_patch_region.top - target.y;

            let ref_patch_region = ref_grid_region.intersection(Region {
                left: patch_ref.x0 as i32 + left,
                top: patch_ref.y0 as i32 + top,
                width,
                height,
            });

            let width = ref_patch_region.width as usize;
            let height = ref_patch_region.height as usize;

            let patch_left = ref_patch_region.left.abs_diff(ref_grid_region.left) as usize;
            let patch_top = ref_patch_region.top.abs_diff(ref_grid_region.top) as usize;
            let base_left = target_patch_region.left.abs_diff(base_grid_region.left) as usize;
            let base_top = target_patch_region.top.abs_diff(base_grid_region.top) as usize;

            let base_topleft = (base_left, base_top);
            let new_topleft = (patch_left, patch_top);

            let bit_depth = if let Some(ec_idx) = idx.checked_sub(color_channels) {
                image_header.metadata.ec_info[ec_idx].bit_depth
            } else {
                image_header.metadata.bit_depth
            };

            let alpha_idx = matches!(
                blending_info.mode,
                PatchBlendMode::BlendAbove
                    | PatchBlendMode::BlendBelow
                    | PatchBlendMode::MulAddAbove
                    | PatchBlendMode::MulAddBelow
            )
            .then_some(blending_info.alpha_channel as usize);

            let base_alpha;
            let new_alpha;
            let premultiplied;
            let base_grid = base_grid.buffer_mut();
            let base_grid = if let Some(alpha_idx) = alpha_idx {
                if alpha_idx + color_channels == idx {
                    base_alpha = None;
                    new_alpha = None;
                    premultiplied = None;
                    &mut base_grid[idx]
                } else {
                    let alpha_bit_depth = image_header.metadata.ec_info[alpha_idx].bit_depth;
                    let (base, alpha) = if idx < alpha_idx + color_channels {
                        let (l, r) = base_grid.split_at_mut(alpha_idx + color_channels);
                        r[0].convert_to_float_modular(alpha_bit_depth)?;
                        (&mut l[idx], &r[0])
                    } else {
                        let (l, r) = base_grid.split_at_mut(idx);
                        l[alpha_idx + color_channels].convert_to_float_modular(alpha_bit_depth)?;
                        (&mut r[0], &l[alpha_idx + color_channels])
                    };
                    base_alpha = Some(alpha.as_float().unwrap().as_subgrid());
                    new_alpha = Some(
                        patch_ref_grid.buffer()[alpha_idx + color_channels]
                            .as_float()
                            .unwrap()
                            .as_subgrid(),
                    );
                    premultiplied = image_header.metadata.ec_info[alpha_idx].alpha_associated();
                    base
                }
            } else {
                base_alpha = None;
                new_alpha = None;
                premultiplied = None;
                &mut base_grid[idx]
            }
            .convert_to_float_modular(bit_depth)?
            .as_subgrid_mut();

            let Some(mut blend_params) = BlendParams::from_patch_blending_info(
                idx,
                color_channels,
                blending_info,
                base_alpha,
                new_alpha,
                premultiplied,
            ) else {
                continue;
            };
            blend_params.base_topleft = base_topleft;
            blend_params.new_topleft = new_topleft;
            blend_params.width = width;
            blend_params.height = height;

            blend_single(
                base_grid,
                patch_ref_grid.buffer()[idx]
                    .as_float()
                    .unwrap()
                    .as_subgrid(),
                &blend_params,
            );
        }
    }

    Ok(())
}

fn blend_single(
    mut base: MutableSubgrid<f32>,
    new_grid: SharedSubgrid<f32>,
    blend_params: &BlendParams<'_>,
) {
    let &BlendParams {
        ref mode,
        base_topleft: (base_x, base_y),
        new_topleft: (new_x, new_y),
        width,
        height,
    } = blend_params;

    match mode {
        BlendMode::Replace | BlendMode::Blend(BlendAlpha { new: None, .. }) => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    base_row[base_buf_x] = new_row[new_buf_x];
                }
            }
        }
        BlendMode::Add | BlendMode::MulAdd(BlendAlpha { new: None, .. }) => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    base_row[base_buf_x] += new_row[new_buf_x];
                }
            }
        }
        BlendMode::Mul(clamp) => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    let mut new_sample = new_row[new_buf_x];
                    if *clamp {
                        new_sample = new_sample.clamp(0.0, 1.0);
                    }
                    base_row[base_buf_x] *= new_sample;
                }
            }
        }
        BlendMode::Blend(BlendAlpha {
            base: base_alpha,
            new: Some(new_alpha),
            clamp,
            swapped,
            premultiplied,
        }) => {
            let base_alpha = base_alpha.as_ref();
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                let base_alpha_row = base_alpha.map(|alpha| alpha.get_row(base_buf_y));
                let new_alpha_row = new_alpha.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_sample;
                    let new_sample;
                    let base_alpha;
                    let mut new_alpha;
                    if *swapped {
                        base_sample = new_row[new_buf_x];
                        new_sample = base_row[base_buf_x];
                        base_alpha = new_alpha_row[new_buf_x];
                        new_alpha = base_alpha_row.map(|b| b[base_buf_x]).unwrap_or(0.0);
                    } else {
                        base_sample = base_row[base_buf_x];
                        new_sample = new_row[new_buf_x];
                        base_alpha = base_alpha_row.map(|b| b[base_buf_x]).unwrap_or(0.0);
                        new_alpha = new_alpha_row[new_buf_x];
                    }

                    if *clamp {
                        new_alpha = new_alpha.clamp(0.0, 1.0);
                    }

                    base_row[base_buf_x] = if *premultiplied {
                        new_sample + base_sample * (1.0 - new_alpha)
                    } else {
                        let base_alpha_rev = 1.0 - base_alpha;
                        let new_alpha_rev = 1.0 - new_alpha;
                        let mixed_alpha = 1.0 - new_alpha_rev * base_alpha_rev;
                        let mixed_alpha_recip = if mixed_alpha > 0.0 {
                            mixed_alpha.recip()
                        } else {
                            0.0
                        };
                        (new_alpha * new_sample + base_alpha * base_sample * new_alpha_rev)
                            * mixed_alpha_recip
                    };
                }
            }
        }
        BlendMode::MulAdd(BlendAlpha {
            base: base_alpha,
            new: Some(new_alpha),
            clamp,
            swapped,
            ..
        }) => {
            let base_alpha = base_alpha.as_ref();
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                let base_alpha_row = base_alpha.map(|alpha| alpha.get_row(base_buf_y));
                let new_alpha_row = new_alpha.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_sample;
                    let new_sample;
                    let mut new_alpha;
                    if *swapped {
                        base_sample = new_row[new_buf_x];
                        new_sample = base_row[base_buf_x];
                        new_alpha = base_alpha_row.map(|b| b[base_buf_x]).unwrap_or(0.0);
                    } else {
                        base_sample = base_row[base_buf_x];
                        new_sample = new_row[new_buf_x];
                        new_alpha = new_alpha_row[new_buf_x];
                    }

                    if *clamp {
                        new_alpha = new_alpha.clamp(0.0, 1.0);
                    }

                    base_row[base_buf_x] = base_sample + new_alpha * new_sample;
                }
            }
        }
        BlendMode::MixAlpha { clamp, swapped } => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                let base_row = base.get_row_mut(base_buf_y);
                let new_row = new_grid.get_row(new_buf_y);
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let mut base = base_row[base_buf_x];
                    let mut new = new_row[new_buf_x];
                    if *swapped {
                        std::mem::swap(&mut base, &mut new);
                    }
                    if *clamp {
                        new = new.clamp(0.0, 1.0);
                    }

                    base_row[base_buf_x] = base + new * (1.0 - base);
                }
            }
        }
        BlendMode::Skip => {}
    }
}
