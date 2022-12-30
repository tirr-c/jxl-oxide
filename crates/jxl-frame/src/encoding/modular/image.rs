use std::io::Read;

use jxl_bitstream::Bitstream;

use crate::{Grid, Result};
use super::{ModularChannels, MaContext, predictor::{SelfCorrectingPredictor, WpHeader}, SubimageChannelInfo};

#[derive(Debug)]
pub struct Image {
    group_dim: u32,
    channels: ModularChannels,
    data: Vec<Grid<i32>>,
}

impl Image {
    pub(super) fn new(channels: ModularChannels, group_dim: u32) -> Self {
        let data = channels.info
            .iter()
            .map(|info| {
                if info.hshift < 0 || info.vshift < 0 {
                    return Grid::new(info.width, info.height, (info.width, info.height));
                }

                let group_dim = if info.hshift >= 3 && info.vshift >= 3 {
                    group_dim * 8
                } else {
                    group_dim
                };
                let gw = group_dim >> info.hshift;
                let gh = group_dim >> info.vshift;
                Grid::new(info.width, info.height, (gw, gh))
            })
            .collect::<Vec<_>>();
        Self {
            group_dim,
            channels,
            data,
        }
    }

    pub fn decode_channels<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        stream_index: u32,
        wp_header: &WpHeader,
        ma_ctx: &mut MaContext,
    ) -> Result<()> {
        let dist_multiplier = self.channels.info.iter()
            .skip(self.channels.nb_meta_channels as usize)
            .map(|info| info.width)
            .max();
        let Some(dist_multiplier) = dist_multiplier else { return Ok(()); };

        let mut channels: Vec<_> = self.channels.info.iter()
            .zip(self.data.iter_mut())
            .enumerate()
            .filter(|(_, (_, grid))| grid.width() != 0 && grid.height() != 0)
            .collect();
        let len = channels.len();
        for idx in 0..len {
            let (prev, left) = channels.split_at_mut(idx);
            let (i, (info, ref mut grid)) = left[0];

            let mut sc_predictor = SelfCorrectingPredictor::new(info, wp_header.clone());
            let width = grid.width();
            let height = grid.height();

            for y in 0..height as i32 {
                let mut prev_prop9 = 0i32;
                for x in 0..width as i32 {
                    let prediction = sc_predictor.predict(grid, x, y);
                    let anchor = grid.anchor(x, y);
                    let mut properties = vec![
                        i as i32,
                        stream_index as i32,
                        y,
                        x,
                        anchor.n().abs(),
                        anchor.w().abs(),
                        anchor.n(),
                        anchor.w(),
                        anchor.w() - prev_prop9,
                        anchor.w() + anchor.n() - anchor.nw(),
                        anchor.w() - anchor.nw(),
                        anchor.nw() - anchor.n(),
                        anchor.n() - anchor.ne(),
                        anchor.n() - anchor.nn(),
                        anchor.w() - anchor.ww(),
                        prediction.max_error,
                    ];
                    prev_prop9 = properties[9];

                    for &mut (_, (prev_info, ref prev_grid)) in prev.iter_mut().rev() {
                        if info.width != prev_info.width ||
                            info.height != prev_info.height ||
                            info.hshift != prev_info.hshift ||
                            info.vshift != prev_info.vshift {
                                continue;
                        }

                        let c = prev_grid[(x, y)];
                        let w = (x > 0).then(|| prev_grid[(x - 1, y)]).unwrap_or(0);
                        let n = (y > 0).then(|| prev_grid[(x, y - 1)]).unwrap_or(w);
                        let nw = (x > 0 && y > 0).then(|| prev_grid[(x - 1, y - 1)]).unwrap_or(w);
                        let g = (w + n - nw).clamp(w.min(n), w.max(n));

                        properties.push(c.abs());
                        properties.push(c);
                        properties.push((c - g).abs());
                        properties.push(c - g);
                    }

                    let (diff, predictor) = ma_ctx.decode_sample(bitstream, &properties, dist_multiplier)?;
                    let sample_prediction = predictor.predict(grid, x, y, &prediction);
                    let true_value = diff + sample_prediction;
                    grid[(x, y)] = true_value;

                    sc_predictor.record_error(prediction, true_value);
                }
            }
        }

        Ok(())
    }
}

impl Image {
    pub(super) fn for_global_modular(&self) -> (Image, Vec<SubimageChannelInfo>) {
        let group_dim = self.group_dim;
        let (channel_info, channel_mapping) = self.channels.info
            .iter()
            .enumerate()
            .filter_map(|(i, info)| {
                if i < self.channels.nb_meta_channels as usize || (info.width <= group_dim && info.height <= group_dim) {
                    Some((info.clone(), SubimageChannelInfo::new(i, 0, 0)))
                } else {
                    None
                }
            })
            .unzip();
        let channels = ModularChannels {
            info: channel_info,
            nb_meta_channels: self.channels.nb_meta_channels,
        };
        (Image::new(channels, group_dim), channel_mapping)
    }

    pub(super) fn copy_from_image(&mut self, subimage: Image, mapping: &[super::SubimageChannelInfo]) -> &mut Self {
        for (grid, mapping) in subimage.data.into_iter().zip(mapping) {
            let SubimageChannelInfo { channel_id, base_x, base_y } = *mapping;
            self.data[channel_id].insert_subgrid(grid, base_x as i32, base_y as i32);
        }
        self
    }
}
