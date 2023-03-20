use jxl_frame::{Frame, header::BlendingInfo};
use jxl_grid::SimpleGrid;
use jxl_image::Headers;

#[derive(Debug)]
enum BlendMode<'a> {
    Replace,
    Add,
    Mul,
    Blend(BlendAlpha<'a>),
    MulAdd(BlendAlpha<'a>),
    MixAlpha,
}

#[derive(Debug)]
struct BlendParams<'a> {
    mode: BlendMode<'a>,
    base_topleft: (usize, usize),
    new_topleft: (usize, usize),
    width: usize,
    height: usize,
}

#[derive(Debug)]
struct BlendAlpha<'a> {
    base: Option<&'a SimpleGrid<f32>>,
    new: &'a SimpleGrid<f32>,
    clamp: bool,
    swapped: bool,
    premultiplied: bool,
}

impl<'a> BlendParams<'a> {
    fn from_blending_info(
        blending_info: &BlendingInfo,
        base_alpha: Option<&'a SimpleGrid<f32>>,
        new_alpha: Option<&'a SimpleGrid<f32>>,
        premultiplied: Option<bool>,
    ) -> Self {
        let mode = match blending_info.mode {
            jxl_frame::header::BlendMode::Replace => BlendMode::Replace,
            jxl_frame::header::BlendMode::Add => BlendMode::Add,
            jxl_frame::header::BlendMode::Blend => {
                BlendMode::Blend(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped: false,
                    premultiplied: premultiplied.unwrap(),
                })
            },
            jxl_frame::header::BlendMode::MulAdd => {
                BlendMode::MulAdd(BlendAlpha {
                    base: base_alpha,
                    new: new_alpha.unwrap(),
                    clamp: blending_info.clamp,
                    swapped: false,
                    premultiplied: premultiplied.unwrap(),
                })
            },
            jxl_frame::header::BlendMode::Mul => BlendMode::Mul,
        };

        Self {
            mode,
            base_topleft: (0, 0),
            new_topleft: (0, 0),
            width: 0,
            height: 0,
        }
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
    tracing::debug!(width, height);

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
            let blend_params = BlendParams {
                mode: BlendMode::MixAlpha,
                base_topleft: (base_x, base_y),
                new_topleft: (new_x, new_y),
                width,
                height,
            };

            output_grid.push(base_grid.clone());
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
            let premultiplied = alpha_idx.map(|idx| image_header.metadata.ec_info[idx].alpha_associated);
            blend_params = BlendParams::from_blending_info(blending_info, base_alpha, new_alpha, premultiplied);
        } else {
            base_grid = &empty_grid;
            let new_alpha = alpha_idx.map(|idx| &new_grid[idx + 3]);
            let premultiplied = alpha_idx.map(|idx| image_header.metadata.ec_info[idx].alpha_associated);
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

fn blend_single(base: &mut SimpleGrid<f32>, new_grid: &SimpleGrid<f32>, blend_params: &BlendParams<'_>) {
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
                    let new_buf_x = new_x + dy;
                    base_buf[base_buf_y * base_stride + base_buf_x] += new_buf[new_buf_y * new_stride + new_buf_x];
                }
            }
        },
        BlendMode::Mul => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dy;
                    base_buf[base_buf_y * base_stride + base_buf_x] *= new_buf[new_buf_y * new_stride + new_buf_x];
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
                    let new_buf_x = new_x + dy;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let new_idx = new_buf_y * new_stride + new_buf_x;

                    let base_sample;
                    let new_sample;
                    if alpha.swapped {
                        base_sample = new_buf[new_idx];
                        new_sample = base_buf[base_idx];
                    } else {
                        base_sample = base_buf[base_idx];
                        new_sample = new_buf[new_idx];
                    }

                    let mut base_alpha = base_alpha_buf.map(|b| b[base_idx]).unwrap_or(0.0);
                    let mut new_alpha = new_alpha_buf[new_idx];
                    if alpha.clamp {
                        base_alpha = base_alpha.clamp(0.0, 1.0);
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
                    let new_buf_x = new_x + dy;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let new_idx = new_buf_y * new_stride + new_buf_x;

                    let base_sample;
                    let new_sample;
                    if alpha.swapped {
                        base_sample = new_buf[new_idx];
                        new_sample = base_buf[base_idx];
                    } else {
                        base_sample = base_buf[base_idx];
                        new_sample = new_buf[new_idx];
                    }

                    let mut base_alpha = base_alpha_buf.map(|b| b[base_idx]).unwrap_or(0.0);
                    let mut new_alpha = new_alpha_buf[new_idx];
                    if alpha.clamp {
                        base_alpha = base_alpha.clamp(0.0, 1.0);
                        new_alpha = new_alpha.clamp(0.0, 1.0);
                    }

                    let mixed_alpha = base_alpha + new_alpha * (1.0 - base_alpha);
                    base_buf[base_idx] = base_sample + mixed_alpha * new_sample;
                }
            }
        },
        BlendMode::MixAlpha => {
            for dy in 0..height {
                let base_buf_y = base_y + dy;
                let new_buf_y = new_y + dy;
                for dx in 0..width {
                    let base_buf_x = base_x + dx;
                    let new_buf_x = new_x + dy;

                    let base_idx = base_buf_y * base_stride + base_buf_x;
                    let base = base_buf[base_idx];
                    base_buf[base_idx] = base + new_buf[new_buf_y * new_stride + new_buf_x] * (1.0 - base);
                }
            }
        },
    }
}
