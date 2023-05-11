use jxl_frame::{
    data::{BlendingModeInformation, PatchRef},
    header::{BlendMode as FrameBlendMode, BlendingInfo},
    Frame,
};
use jxl_grid::SimpleGrid;
use jxl_image::Headers;

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
    image_header: &Headers,
    reference_grids: [Option<&[SimpleGrid<f32>]>; 4],
    new_frame: &Frame<'_>,
    new_grid: &[SimpleGrid<f32>],
) -> Vec<SimpleGrid<f32>> {
    let header = new_frame.header();

    let base_x = header.x0.max(0) as usize;
    let base_y = header.y0.max(0) as usize;
    let new_x = (-header.x0).max(0) as usize;
    let new_y = (-header.y0).max(0) as usize;

    let base_width = image_header.size.width as usize;
    let base_height = image_header.size.height as usize;
    let new_width = new_grid[0].width();
    let new_height = new_grid[0].height();
    let out_of_range = base_width <= base_x ||
        base_height <= base_y ||
        new_width <= new_x ||
        new_height <= new_y;

    let width;
    let height;
    if out_of_range {
        width = 0;
        height = 0;
    } else {
        width = (base_width - base_x).min(new_width - new_x);
        height = (base_height - base_y).min(new_height - new_y);
    }

    let mut used_as_alpha = vec![false; 3 + image_header.metadata.num_extra as usize];
    for blending_info in std::iter::once(&header.blending_info).chain(&header.ec_blending_info) {
        if let (_, Some(alpha)) = source_and_alpha_from_blending_info(blending_info) {
            used_as_alpha[alpha + 3] = true;
        }
    }

    let empty_grid = SimpleGrid::new(base_width, base_height);
    let mut output_grid = Vec::with_capacity(3 + image_header.metadata.num_extra as usize);
    for (idx, blending_info) in [&header.blending_info; 3].into_iter().chain(&header.ec_blending_info).enumerate() {
        let (ref_idx, alpha_idx) = source_and_alpha_from_blending_info(blending_info);
        let ref_grid = reference_grids[ref_idx];

        if used_as_alpha[idx] {
            let base_grid = if let Some(grid) = ref_grid {
                &grid[idx]
            } else {
                &empty_grid
            };
            output_grid.push(base_grid.clone());

            if blending_info.mode == FrameBlendMode::MulAdd {
                continue;
            }

            let blend_params = BlendParams {
                mode: BlendMode::MixAlpha(blending_info.clamp),
                base_topleft: (base_x, base_y),
                new_topleft: (new_x, new_y),
                width,
                height,
            };

            let base_grid = &mut output_grid[idx];
            blend_single(base_grid, &new_grid[idx], &blend_params);
            continue;
        }

        let base_grid;
        let mut blend_params;
        if let Some(grid) = ref_grid {
            base_grid = &grid[idx];
            let base_alpha = alpha_idx.map(|idx| &grid[idx + 3]);
            let new_alpha = alpha_idx.map(|idx| &new_grid[idx + 3]);
            let premultiplied = alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            blend_params = BlendParams::from_blending_info(blending_info, base_alpha, new_alpha, premultiplied);
        } else {
            base_grid = &empty_grid;
            let new_alpha = alpha_idx.map(|idx| &new_grid[idx + 3]);
            let premultiplied = alpha_idx.and_then(|idx| image_header.metadata.ec_info[idx].alpha_associated());
            blend_params = BlendParams::from_blending_info(blending_info, None, new_alpha, premultiplied);
        }
        blend_params.base_topleft = (base_x, base_y);
        blend_params.new_topleft = (new_x, new_y);
        blend_params.width = width;
        blend_params.height = height;

        output_grid.push(base_grid.clone());
        let base_grid = &mut output_grid[idx];
        blend_single(base_grid, &new_grid[idx], &blend_params);
    }

    output_grid
}

pub fn patch(
    image_header: &Headers,
    base_grid: &mut [SimpleGrid<f32>],
    patch_ref_grid: &[SimpleGrid<f32>],
    patch_ref: &PatchRef,
) {
    use jxl_frame::data::PatchBlendMode;

    let new_x = patch_ref.x0 as usize;
    let new_y = patch_ref.y0 as usize;
    let base_width = base_grid[0].width();
    let base_height = base_grid[0].height();
    let new_width = patch_ref.width as usize;
    let new_height = patch_ref.height as usize;

    for target in &patch_ref.patch_targets {
        let (new_x, new_width) = if target.x < 0 {
            let abs = target.x.unsigned_abs() as usize;
            (new_x + abs, new_width.saturating_sub(abs))
        } else {
            (new_x, new_width)
        };
        let (new_y, new_height) = if target.y < 0 {
            let abs = target.y.unsigned_abs() as usize;
            (new_y + abs, new_height.saturating_sub(abs))
        } else {
            (new_y, new_height)
        };

        let base_x = target.x.max(0) as usize;
        let base_y = target.y.max(0) as usize;
        let out_of_range = base_width <= base_x || base_height <= base_y;

        let width;
        let height;
        if out_of_range {
            width = 0;
            height = 0;
        } else {
            width = (base_width - base_x).min(new_width);
            height = (base_height - base_y).min(new_height);
        }

        let mut used_as_alpha = vec![false; 3 + image_header.metadata.num_extra as usize];
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
                        base_topleft: (base_x, base_y),
                        new_topleft: (new_x, new_y),
                        width,
                        height,
                    };
                    blend_single(&mut base_grid[idx], &patch_ref_grid[idx], &blend_params);
                    continue;
                }

                let blend_params = BlendParams {
                    mode: BlendMode::MixAlpha(blending_info.clamp),
                    base_topleft: (base_x, base_y),
                    new_topleft: (new_x, new_y),
                    width,
                    height,
                };
                blend_single(&mut base_grid[idx], &patch_ref_grid[idx], &blend_params);
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
            let base_grid = if let Some(alpha_idx) = alpha_idx {
                let (base, alpha) = if idx < alpha_idx + 3 {
                    let (l, r) = base_grid.split_at_mut(alpha_idx + 3);
                    (&mut l[idx], &r[0])
                } else {
                    let (l, r) = base_grid.split_at_mut(idx);
                    (&mut r[0], &l[alpha_idx + 3])
                };
                base_alpha = Some(alpha);
                new_alpha = Some(&patch_ref_grid[alpha_idx + 3]);
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
            blend_params.base_topleft = (base_x, base_y);
            blend_params.new_topleft = (new_x, new_y);
            blend_params.width = width;
            blend_params.height = height;

            blend_single(base_grid, &patch_ref_grid[idx], &blend_params);
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
