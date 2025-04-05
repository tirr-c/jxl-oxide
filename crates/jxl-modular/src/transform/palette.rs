use jxl_grid::{MutableSubgrid, SharedSubgrid};

use crate::{
    Sample,
    predictor::{Predictor, PredictorState},
};

use super::Palette;

#[rustfmt::skip]
const DELTA_PALETTE: [[i16; 3]; 72] = [
    [0, 0, 0], [4, 4, 4], [11, 0, 0], [0, 0, -13], [0, -12, 0], [-10, -10, -10],
    [-18, -18, -18], [-27, -27, -27], [-18, -18, 0], [0, 0, -32], [-32, 0, 0], [-37, -37, -37],
    [0, -32, -32], [24, 24, 45], [50, 50, 50], [-45, -24, -24], [-24, -45, -45], [0, -24, -24],
    [-34, -34, 0], [-24, 0, -24], [-45, -45, -24], [64, 64, 64], [-32, 0, -32], [0, -32, 0],
    [-32, 0, 32], [-24, -45, -24], [45, 24, 45], [24, -24, -45], [-45, -24, 24], [80, 80, 80],
    [64, 0, 0], [0, 0, -64], [0, -64, -64], [-24, -24, 45], [96, 96, 96], [64, 64, 0],
    [45, -24, -24], [34, -34, 0], [112, 112, 112], [24, -45, -45], [45, 45, -24], [0, -32, 32],
    [24, -24, 45], [0, 96, 96], [45, -24, 24], [24, -45, -24], [-24, -45, 24], [0, -64, 0],
    [96, 0, 0], [128, 128, 128], [64, 0, 64], [144, 144, 144], [96, 96, 0], [-36, -36, 36],
    [45, -24, -45], [45, -45, -24], [0, 0, -96], [0, 128, 128], [0, 96, 0], [45, 24, -45],
    [-128, 0, 0], [24, -45, 24], [-45, 24, -45], [64, 0, -64], [64, -64, -64], [96, 0, 96],
    [45, -45, 24], [24, 45, -45], [64, 64, -64], [128, 128, 0], [0, 0, -128], [-24, 45, -45],
];

impl Palette {
    pub(crate) fn inverse_inner<S: Sample>(
        &self,
        palette: SharedSubgrid<S>,
        mut targets: Vec<MutableSubgrid<S>>,
        bit_depth: u32,
    ) {
        let nb_deltas = self.nb_deltas as i32;
        let nb_colors = self.nb_colours as i32;

        let is_simple = {
            let index_grid = targets[0].as_shared();
            let height = index_grid.height();
            (0..height).all(|y| {
                let row = index_grid.get_row(y);
                row.iter()
                    .all(|&index| (0..nb_colors).contains(&index.to_i32()))
            })
        };

        if is_simple {
            return inverse_simple(palette, targets);
        }

        tracing::trace!("Inverse palette, slow path");

        let mut need_delta = Vec::new();
        let width = targets[0].width();
        let height = targets[0].height();
        let channels = targets.len();
        assert_eq!(channels, palette.height());
        for y in 0..height {
            for x in 0..width {
                let index = targets[0].get(x, y).to_i32();
                if index < nb_deltas {
                    need_delta.push((x, y));
                }

                let channels_it = targets.iter_mut().map(|g| g.get_mut(x, y));

                if (0..nb_colors).contains(&index) {
                    for (c, sample) in channels_it.enumerate() {
                        *sample = *palette.get(index as usize, c);
                    }
                } else if index >= nb_colors {
                    let index = index - nb_colors;
                    if index < 64 {
                        for (c, sample) in channels_it.enumerate() {
                            *sample = S::from_i32(
                                ((index >> (2 * c)) % 4) * ((1i32 << bit_depth) - 1) / 4
                                    + (1i32 << bit_depth.saturating_sub(3)),
                            );
                        }
                    } else {
                        let mut index = index - 64;
                        for sample in channels_it {
                            *sample = S::from_i32((index % 5) * ((1i32 << bit_depth) - 1) / 4);
                            index /= 5;
                        }
                    }
                } else {
                    for (c, sample) in channels_it.enumerate() {
                        if c >= 3 {
                            *sample = S::default();
                            continue;
                        }

                        let index = -(index + 1);
                        let index = (index % 143) as usize;
                        let mut temp_sample = DELTA_PALETTE[(index + 1) >> 1][c] as i32;
                        if index & 1 == 0 {
                            temp_sample = -temp_sample;
                        }
                        if bit_depth > 8 {
                            temp_sample <<= bit_depth.min(24) - 8;
                        }
                        *sample = S::from_i32(temp_sample);
                    }
                }
            }
        }

        if need_delta.is_empty() {
            return;
        }

        let d_pred = self.d_pred;
        let wp_header = if d_pred == Predictor::SelfCorrecting {
            self.wp_header.as_ref()
        } else {
            None
        };
        let mut predictor = PredictorState::<S>::new();

        'outer: for mut grid in targets {
            predictor.reset(width as u32, &[], wp_header);

            let mut idx = 0;
            for y in 0..height {
                for x in 0..width {
                    let properties = predictor.properties::<true>();
                    let sample = grid.get_mut(x, y);
                    let mut sample_value = sample.to_i32();
                    if need_delta[idx] == (x, y) {
                        let diff = d_pred.predict::<_, true>(&properties);
                        sample_value = sample_value.wrapping_add(diff);
                        *sample = S::from_i32(sample_value);
                        idx += 1;
                        if idx >= need_delta.len() {
                            continue 'outer;
                        }
                    }
                    properties.record(sample_value);
                }
            }
        }
    }
}

#[inline(never)]
fn inverse_simple<S: Sample>(palette: SharedSubgrid<S>, targets: Vec<MutableSubgrid<S>>) {
    let height = targets[0].height();
    let channels = targets.len();
    assert_eq!(channels, palette.height());

    tracing::trace!("Inverse palette, fast path");

    let mut targets_it = targets.into_iter().enumerate();
    let (_, mut index_grid) = targets_it.next().unwrap();
    for (c, mut grid) in targets_it {
        let palette = palette.get_row(c);
        for y in 0..height {
            let index_row = index_grid.get_row(y);
            let grid_row = grid.get_row_mut(y);
            for (index, sample) in index_row.iter().zip(grid_row) {
                *sample = palette[index.to_i32() as usize];
            }
        }
    }

    let palette = palette.get_row(0);
    for y in 0..height {
        let grid_row = index_grid.get_row_mut(y);
        for sample in grid_row {
            *sample = palette[sample.to_i32() as usize];
        }
    }
}
