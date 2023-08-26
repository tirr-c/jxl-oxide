use jxl_frame::{
    data::{BlendingModeInformation, PatchRef},
    header::{BlendMode as FrameBlendMode, BlendingInfo},
    Frame,
};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;

use crate::{region::{ImageWithRegion, Region}, inner::Reference};

#[derive(Debug)]
enum BlendMode<'a> {
    Replace,
    Add,
    Mul(bool),
    Blend(BlendAlpha<'a>),
    MulAdd(BlendAlpha<'a>),
    MixAlpha(bool),
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
        blending_info: &BlendingInfo,
        base_alpha: Option<&'a SimpleGrid<f32>>,
        new_alpha: Option<&'a SimpleGrid<f32>>,
        premultiplied: Option<bool>,
    ) -> Self {
        let mode = match blending_info.mode {
            FrameBlendMode::Replace => BlendMode::Replace,
            FrameBlendMode::Add => BlendMode::Add,
            FrameBlendMode::Blend => {
                BlendMode::Blend(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped: false,
                    premultiplied: premultiplied.unwrap(),
                })
            },
            FrameBlendMode::MulAdd => {
                BlendMode::MulAdd(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped: false,
                    premultiplied: premultiplied.unwrap(),
                })
            },
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
                BlendMode::Blend(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped,
                    premultiplied: premultiplied.unwrap(),
                })
            },
            PatchBlendMode::MulAddAbove | PatchBlendMode::MulAddBelow => {
                let swapped = blending_info.mode == PatchBlendMode::MulAddBelow;
                BlendMode::MulAdd(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped,
                    premultiplied: premultiplied.unwrap(),
                })
            },
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

pub fn blend(
    image_header: &ImageHeader,
    reference_grids: [Option<Reference>; 4],
    new_frame: &Frame,
    new_grid: &ImageWithRegion,
) -> ImageWithRegion {
    let header = new_frame.header();
    let channels = 3 + image_header.metadata.ec_info.len();
    let mut output_grid = ImageWithRegion::from_region(channels, new_grid.region());
    let output_image_region = new_grid.region().translate(header.x0, header.y0);

    let mut used_as_alpha = vec![false; 3 + image_header.metadata.ec_info.len()];
    for blending_info in std::iter::once(&header.blending_info).chain(&header.ec_blending_info) {
        if let (_, Some(alpha)) = source_and_alpha_from_blending_info(blending_info) {
            used_as_alpha[alpha + 3] = true;
        }
    }

    for (idx, blending_info) in [&header.blending_info; 3].into_iter().chain(&header.ec_blending_info).enumerate() {
        let (ref_idx, alpha_idx) = source_and_alpha_from_blending_info(blending_info);
        let ref_grid = reference_grids[ref_idx];

        if used_as_alpha[idx] {
            let empty_image;
            let (base_grid, base_frame_region) = if let Some(grid) = ref_grid {
                let base_frame_header = grid.frame.header();
                let base_frame_region = output_image_region.translate(-base_frame_header.x0, -base_frame_header.y0);
                (grid.image, base_frame_region)
            } else {
                empty_image = ImageWithRegion::from_region(channels, Region::empty());
                (&empty_image, Region::empty())
            };
            base_grid.clone_region_channel(base_frame_region, idx, &mut output_grid.buffer_mut()[idx]);

            if blending_info.mode == FrameBlendMode::MulAdd {
                continue;
            }

            let blend_params = BlendParams {
                mode: BlendMode::MixAlpha(blending_info.clamp),
                base_topleft: (0, 0),
                new_topleft: (0, 0),
                width: output_image_region.width as usize,
                height: output_image_region.height as usize,
            };

            let base_grid = &mut output_grid.buffer_mut()[idx];
            blend_single(base_grid, &new_grid.buffer()[idx], &blend_params);
            continue;
        }

        let mut base_alpha_grid;
        let mut blend_params = if let Some(grid) = ref_grid {
            let base_frame_header = grid.frame.header();
            let base_frame_region = output_image_region.translate(-base_frame_header.x0, -base_frame_header.y0);
            grid.image.clone_region_channel(base_frame_region, idx, &mut output_grid.buffer_mut()[idx]);

            let base_alpha = if let Some(idx) = alpha_idx {
                base_alpha_grid = SimpleGrid::new(output_image_region.width as usize, output_image_region.height as usize);
                grid.image.clone_region_channel(base_frame_region, idx + 3, &mut base_alpha_grid);
                Some(&base_alpha_grid)
            } else {
                None
            };
            let new_alpha = alpha_idx.map(|idx| &new_grid.buffer()[idx + 3]);
            let premultiplied = alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(blending_info, base_alpha, new_alpha, premultiplied)
        } else {
            let new_alpha = alpha_idx.map(|idx| &new_grid.buffer()[idx + 3]);
            let premultiplied = alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            BlendParams::from_blending_info(blending_info, None, new_alpha, premultiplied)
        };
        blend_params.width = output_image_region.width as usize;
        blend_params.height = output_image_region.height as usize;

        let base_grid = &mut output_grid.buffer_mut()[idx];
        blend_single(base_grid, &new_grid.buffer()[idx], &blend_params);
    }

    output_grid
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

        let mut used_as_alpha = vec![false; 3 + image_header.metadata.ec_info.len()];
        for blending_info in &target.blending {
            if matches!(
                blending_info.mode,
                | PatchBlendMode::BlendAbove
                | PatchBlendMode::BlendBelow
                | PatchBlendMode::MulAddAbove
                | PatchBlendMode::MulAddBelow
            ) {
                used_as_alpha[blending_info.alpha_channel as usize + 3] = true;
            }
        }

        for (idx, blending_info) in [&target.blending[0]; 3].into_iter().chain(&target.blending[1..]).enumerate() {
            if used_as_alpha[idx] {
                if blending_info.mode == PatchBlendMode::MulAddAbove {
                    continue;
                }
                if blending_info.mode == PatchBlendMode::MulAddBelow {
                    let blend_params = BlendParams {
                        mode: BlendMode::Replace,
                        base_topleft,
                        new_topleft,
                        width,
                        height,
                    };
                    blend_single(&mut base_grid.buffer_mut()[idx], &patch_ref_grid.buffer()[idx], &blend_params);
                    continue;
                }

                let blend_params = BlendParams {
                    mode: BlendMode::MixAlpha(blending_info.clamp),
                    base_topleft,
                    new_topleft,
                    width,
                    height,
                };
                blend_single(&mut base_grid.buffer_mut()[idx], &patch_ref_grid.buffer()[idx], &blend_params);
                continue;
            }

            let alpha_idx = matches!(
                blending_info.mode,
                | PatchBlendMode::BlendAbove
                | PatchBlendMode::BlendBelow
                | PatchBlendMode::MulAddAbove
                | PatchBlendMode::MulAddBelow
            ).then_some(blending_info.alpha_channel as usize);
            let base_alpha;
            let new_alpha;
            let premultiplied;
            let base_grid = base_grid.buffer_mut();
            let base_grid = if let Some(alpha_idx) = alpha_idx {
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
            } else {
                base_alpha = None;
                new_alpha = None;
                premultiplied = None;
                &mut base_grid[idx]
            };

            let Some(mut blend_params) = BlendParams::from_patch_blending_info(
                blending_info,
                base_alpha,
                new_alpha,
                premultiplied,
            ) else { continue; };
            blend_params.base_topleft = base_topleft;
            blend_params.new_topleft = new_topleft;
            blend_params.width = width;
            blend_params.height = height;

            blend_single(base_grid, &patch_ref_grid.buffer()[idx], &blend_params);
        }
    }
}

fn blend_single(base: &mut SimpleGrid<f32>, new_grid: &SimpleGrid<f32>, blend_params: &BlendParams<'_>) {
    tracing::trace!(blend_params = format_args!("{:?}", blend_params));
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
                    base_buf[base_buf_y * base_stride + base_buf_x] = new_buf[new_buf_y * new_stride + new_buf_x];
                }
            }
        },
        BlendMode::Add => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;
                    base_buf[base_buf_y * base_stride + base_buf_x] += new_buf[new_buf_y * new_stride + new_buf_x];
                }
            }
        },
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
        },
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
                        (new_alpha * new_sample + base_alpha * base_sample * (1.0 - new_alpha)) / mixed_alpha
                    };
                }
            }
        },
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
        },
        BlendMode::MixAlpha(clamp) => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dx;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let base = base_buf[base_idx];
                    let mut new = new_buf[new_buf_y * new_stride + new_buf_x];
                    if *clamp {
                        new = new.clamp(0.0, 1.0);
                    }

                    base_buf[base_idx] = base + new * (1.0 - base);
                }
            }
        },
    }
}
