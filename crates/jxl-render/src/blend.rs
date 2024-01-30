use std::sync::Arc;

use jxl_frame::{
    data::{BlendingModeInformation, PatchRef},
    header::{BlendMode as FrameBlendMode, BlendingInfo},
    Frame,
};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, Reference, Region, Result};

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
    base: Option<&'a SimpleGrid<f32>>,
    new: &'a SimpleGrid<f32>,
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
        blending_info: &BlendingInfo,
        base_alpha: Option<&'a SimpleGrid<f32>>,
        new_alpha: Option<&'a SimpleGrid<f32>>,
        premultiplied: Option<bool>,
    ) -> Self {
        let mode = match blending_info.mode {
            FrameBlendMode::Replace => BlendMode::Replace,
            FrameBlendMode::Add => BlendMode::Add,
            FrameBlendMode::Blend if channel_idx == blending_info.alpha_channel as usize + 3 => {
                BlendMode::MixAlpha {
                    clamp: blending_info.clamp,
                    swapped: false,
                }
            }
            FrameBlendMode::Blend => BlendMode::Blend(BlendAlpha {
                base: base_alpha,
                new: new_alpha.unwrap(),
                clamp: blending_info.clamp,
                swapped: false,
                premultiplied: premultiplied.unwrap(),
            }),
            FrameBlendMode::MulAdd if channel_idx == blending_info.alpha_channel as usize + 3 => {
                BlendMode::Skip
            }
            FrameBlendMode::MulAdd => BlendMode::MulAdd(BlendAlpha {
                base: base_alpha,
                new: new_alpha.unwrap(),
                clamp: blending_info.clamp,
                swapped: false,
                premultiplied: premultiplied.unwrap(),
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
        blending_info: &BlendingModeInformation,
        base_alpha: Option<&'a SimpleGrid<f32>>,
        new_alpha: Option<&'a SimpleGrid<f32>>,
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
                if channel_idx == blending_info.alpha_channel as usize + 3 {
                    BlendMode::MixAlpha {
                        clamp: blending_info.clamp,
                        swapped,
                    }
                } else {
                    BlendMode::Blend(BlendAlpha {
                        base: base_alpha,
                        new: new_alpha.unwrap(),
                        clamp: blending_info.clamp,
                        swapped,
                        premultiplied: premultiplied.unwrap(),
                    })
                }
            }
            PatchBlendMode::MulAddAbove | PatchBlendMode::MulAddBelow => {
                let swapped = blending_info.mode == PatchBlendMode::MulAddBelow;
                if channel_idx == blending_info.alpha_channel as usize + 3 {
                    if swapped {
                        BlendMode::Replace
                    } else {
                        BlendMode::Skip
                    }
                } else {
                    BlendMode::MulAdd(BlendAlpha {
                        base: base_alpha,
                        new: new_alpha.unwrap(),
                        clamp: blending_info.clamp,
                        swapped,
                        premultiplied: premultiplied.unwrap(),
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

fn source_and_alpha_from_blending_info(blending_info: &BlendingInfo) -> (usize, Option<usize>) {
    use jxl_frame::header::BlendMode;

    let source = blending_info.source as usize;
    let alpha = matches!(blending_info.mode, BlendMode::Blend | BlendMode::MulAdd)
        .then_some(blending_info.alpha_channel as usize);

    (source, alpha)
}

pub(crate) fn blend<S: Sample>(
    image_header: &ImageHeader,
    reference_grids: [Option<Reference<S>>; 4],
    new_frame: &Frame,
    new_grid: &ImageWithRegion,
    output_frame_region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let header = new_frame.header();
    let tracker = new_frame.alloc_tracker();

    let original_frame_region = new_grid.region();
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

    let mut output_grid = ImageWithRegion::from_region_and_tracker(
        0,
        output_frame_region,
        new_grid.ct_done(),
        tracker,
    )?;

    for (idx, blending_info) in [&header.blending_info; 3]
        .into_iter()
        .chain(&header.ec_blending_info)
        .enumerate()
    {
        let (ref_idx, alpha_idx) = source_and_alpha_from_blending_info(blending_info);
        let ref_grid = &reference_grids[ref_idx];
        let mut can_overwrite = idx < 3
            && (header.is_last
                || (header.can_reference() && ref_idx == header.save_as_reference as usize));

        let clipped_original_frame_region = original_frame_region.intersection(output_frame_region);

        let mut base_alpha_grid;
        let mut base_alpha = None;
        let mut target_grid;
        let mut clone_empty = false;
        if let Some(grid) = ref_grid {
            if grid.frame.header().is_keyframe() {
                can_overwrite = false;
            }

            let base_grid = Arc::clone(&grid.image).run_with_image()?;
            let mut base_grid = base_grid.blend(Some(output_image_region), pool)?;

            if base_grid.region().is_empty() {
                clone_empty = true;
                target_grid = SimpleGrid::with_alloc_tracker(
                    output_frame_region.width as usize,
                    output_frame_region.height as usize,
                    tracker,
                )?;
            } else {
                let base_frame_header = grid.frame.header();
                let base_frame_region =
                    output_image_region.translate(-base_frame_header.x0, -base_frame_header.y0);
                target_grid = if base_grid.region() == base_frame_region && can_overwrite {
                    std::mem::replace(
                        &mut base_grid.buffer_mut()[idx],
                        SimpleGrid::with_alloc_tracker(0, 0, tracker)?,
                    )
                } else {
                    let mut output_grid = SimpleGrid::with_alloc_tracker(
                        output_frame_region.width as usize,
                        output_frame_region.height as usize,
                        tracker,
                    )?;
                    base_grid.clone_region_channel(base_frame_region, idx, &mut output_grid);
                    output_grid
                };

                if let Some(alpha_idx) = alpha_idx {
                    if alpha_idx + 3 != idx {
                        base_alpha_grid = SimpleGrid::with_alloc_tracker(
                            output_frame_region.width as usize,
                            output_frame_region.height as usize,
                            tracker,
                        )?;
                        base_grid.clone_region_channel(
                            base_frame_region,
                            alpha_idx + 3,
                            &mut base_alpha_grid,
                        );
                        base_alpha = Some(&base_alpha_grid)
                    }
                }
            }
        } else {
            clone_empty = true;
            target_grid = SimpleGrid::with_alloc_tracker(
                output_frame_region.width as usize,
                output_frame_region.height as usize,
                tracker,
            )?;
        }

        let mut blend_params = if clone_empty {
            let new_alpha = alpha_idx.map(|idx| &new_grid.buffer()[idx + 3]);
            let premultiplied =
                alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(idx, blending_info, None, new_alpha, premultiplied)
        } else {
            let new_alpha = alpha_idx.map(|idx| &new_grid.buffer()[idx + 3]);
            let premultiplied =
                alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(
                idx,
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

        blend_single(&mut target_grid, &new_grid.buffer()[idx], &blend_params);
        output_grid.push_channel(target_grid);
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
) {
    use jxl_frame::data::PatchBlendMode;

    let base_grid_region = base_grid.region();
    let ref_grid_region = patch_ref_grid.region();

    for target in &patch_ref.patch_targets {
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

        for (idx, blending_info) in [&target.blending[0]; 3]
            .into_iter()
            .chain(&target.blending[1..])
            .enumerate()
        {
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
                if alpha_idx + 3 == idx {
                    base_alpha = None;
                    new_alpha = None;
                    premultiplied = None;
                    &mut base_grid[idx]
                } else {
                    let (base, alpha) = if idx < alpha_idx + 3 {
                        let (l, r) = base_grid.split_at_mut(alpha_idx + 3);
                        (&mut l[idx], &r[0])
                    } else {
                        let (l, r) = base_grid.split_at_mut(idx);
                        (&mut r[0], &l[alpha_idx + 3])
                    };
                    base_alpha = Some(alpha);
                    new_alpha = Some(&patch_ref_grid.buffer()[alpha_idx + 3]);
                    premultiplied = image_header.metadata.ec_info[alpha_idx].alpha_associated();
                    base
                }
            } else {
                base_alpha = None;
                new_alpha = None;
                premultiplied = None;
                &mut base_grid[idx]
            };

            let Some(mut blend_params) = BlendParams::from_patch_blending_info(
                idx,
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

            blend_single(base_grid, &patch_ref_grid.buffer()[idx], &blend_params);
        }
    }
}

fn blend_single(
    base: &mut SimpleGrid<f32>,
    new_grid: &SimpleGrid<f32>,
    blend_params: &BlendParams<'_>,
) {
    let &BlendParams {
        ref mode,
        base_topleft: (base_x, base_y),
        new_topleft: (new_x, new_y),
        width,
        height,
    } = blend_params;

    let base_stride = base.width();
    let new_stride = new_grid.width();
    let base_buf = base.buf_mut();
    let new_buf = new_grid.buf();

    match mode {
        BlendMode::Replace => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    base_buf[base_buf_y * base_stride + base_buf_x] =
                        new_buf[new_buf_y * new_stride + new_buf_x];
                }
            }
        }
        BlendMode::Add => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    base_buf[base_buf_y * base_stride + base_buf_x] +=
                        new_buf[new_buf_y * new_stride + new_buf_x];
                }
            }
        }
        BlendMode::Mul(clamp) => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    let mut new_sample = new_buf[new_buf_y * new_stride + new_buf_x];
                    if *clamp {
                        new_sample = new_sample.clamp(0.0, 1.0);
                    }
                    base_buf[base_buf_y * base_stride + base_buf_x] *= new_sample;
                }
            }
        }
        BlendMode::Blend(alpha) => {
            let base_alpha_buf = alpha.base.map(|g| g.buf());
            let new_alpha_buf = alpha.new.buf();
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let new_idx = new_buf_y * new_stride + new_buf_x;

                    let base_sample;
                    let new_sample;
                    let base_alpha;
                    let mut new_alpha;
                    if alpha.swapped {
                        base_sample = new_buf[new_idx];
                        new_sample = base_buf[base_idx];
                        base_alpha = new_alpha_buf[new_idx];
                        new_alpha = base_alpha_buf.map(|b| b[base_idx]).unwrap_or(0.0);
                    } else {
                        base_sample = base_buf[base_idx];
                        new_sample = new_buf[new_idx];
                        base_alpha = base_alpha_buf.map(|b| b[base_idx]).unwrap_or(0.0);
                        new_alpha = new_alpha_buf[new_idx];
                    }

                    if alpha.clamp {
                        new_alpha = new_alpha.clamp(0.0, 1.0);
                    }

                    base_buf[base_idx] = if alpha.premultiplied {
                        new_sample + base_sample * (1.0 - new_alpha)
                    } else {
                        let mixed_alpha = base_alpha + new_alpha * (1.0 - base_alpha);
                        (new_alpha * new_sample + base_alpha * base_sample * (1.0 - new_alpha))
                            / mixed_alpha
                    };
                }
            }
        }
        BlendMode::MulAdd(alpha) => {
            let base_alpha_buf = alpha.base.map(|g| g.buf());
            let new_alpha_buf = alpha.new.buf();
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let new_idx = new_buf_y * new_stride + new_buf_x;

                    let base_sample;
                    let new_sample;
                    let mut new_alpha;
                    if alpha.swapped {
                        base_sample = new_buf[new_idx];
                        new_sample = base_buf[base_idx];
                        new_alpha = base_alpha_buf.map(|b| b[base_idx]).unwrap_or(0.0);
                    } else {
                        base_sample = base_buf[base_idx];
                        new_sample = new_buf[new_idx];
                        new_alpha = new_alpha_buf[new_idx];
                    }

                    if alpha.clamp {
                        new_alpha = new_alpha.clamp(0.0, 1.0);
                    }

                    base_buf[base_idx] = base_sample + new_alpha * new_sample;
                }
            }
        }
        BlendMode::MixAlpha { clamp, swapped } => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let mut base = base_buf[base_idx];
                    let mut new = new_buf[new_buf_y * new_stride + new_buf_x];
                    if *swapped {
                        std::mem::swap(&mut base, &mut new);
                    }
                    if *clamp {
                        new = new.clamp(0.0, 1.0);
                    }

                    base_buf[base_idx] = base + new * (1.0 - base);
                }
            }
        }
        BlendMode::Skip => {}
    }
}
