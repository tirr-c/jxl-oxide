use std::cell::Cell;

use jxl_bitstream::{define_bundle, read_bits, Bitstream};

use crate::{Error, Result};

define_bundle! {
    #[derive(Debug, Clone)]
    pub struct WpHeader error(crate::Error) {
        default_wp: ty(Bool) default(true),
        wp_p1: ty(u(5)) cond(!default_wp) default(16),
        wp_p2: ty(u(5)) cond(!default_wp) default(10),
        wp_p3a: ty(u(5)) cond(!default_wp) default(7),
        wp_p3b: ty(u(5)) cond(!default_wp) default(7),
        wp_p3c: ty(u(5)) cond(!default_wp) default(7),
        wp_p3d: ty(u(5)) cond(!default_wp) default(0),
        wp_p3e: ty(u(5)) cond(!default_wp) default(0),
        wp_w0: ty(u(4)) cond(!default_wp) default(13),
        wp_w1: ty(u(4)) cond(!default_wp) default(12),
        wp_w2: ty(u(4)) cond(!default_wp) default(12),
        wp_w3: ty(u(4)) cond(!default_wp) default(12),
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(u8)]
pub enum Predictor {
    #[default]
    Zero = 0,
    West,
    North,
    AvgWestAndNorth,
    Select,
    Gradient,
    SelfCorrecting,
    NorthEast,
    NorthWest,
    WestWest,
    AvgWestAndNorthWest,
    AvgNorthAndNorthWest,
    AvgNorthAndNorthEast,
    AvgAll,
}

impl TryFrom<u32> for Predictor {
    type Error = jxl_bitstream::Error;

    fn try_from(value: u32) -> jxl_bitstream::Result<Self> {
        use Predictor::*;
        Ok(match value {
            0 => Zero,
            1 => West,
            2 => North,
            3 => AvgWestAndNorth,
            4 => Select,
            5 => Gradient,
            6 => SelfCorrecting,
            7 => NorthEast,
            8 => NorthWest,
            9 => WestWest,
            10 => AvgWestAndNorthWest,
            11 => AvgNorthAndNorthWest,
            12 => AvgNorthAndNorthEast,
            13 => AvgAll,
            _ => return Err(jxl_bitstream::Error::InvalidEnum { name: "MaProperty", value }),
        })
    }
}

impl Predictor {
    pub(super) fn predict(self, properties: &Properties<'_, '_>) -> i32 {
        use Predictor::*;
        let predictor = &*properties.predictor;

        match self {
            Zero => 0,
            West => predictor.w(),
            North => predictor.n(),
            AvgWestAndNorth => (predictor.w() + predictor.n()) / 2,
            Select => {
                let n = predictor.n();
                let w = predictor.w();
                let nw = predictor.nw();
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            },
            Gradient => {
                let n = predictor.n();
                let w = predictor.w();
                let nw = predictor.nw();
                (w + n - nw).clamp(w.min(n), w.max(n))
            },
            SelfCorrecting => {
                let prediction = properties
                    .prediction()
                    .expect("predict_non_sc called with SelfCorrecting predictor");
                (prediction + 3) >> 3
            },
            NorthEast => predictor.ne(),
            NorthWest => predictor.nw(),
            WestWest => predictor.ww(),
            AvgWestAndNorthWest => (predictor.w() + predictor.nw()) / 2,
            AvgNorthAndNorthWest => (predictor.n() + predictor.nw()) / 2,
            AvgNorthAndNorthEast => (predictor.n() + predictor.ne()) / 2,
            AvgAll => {
                let n = predictor.n();
                let w = predictor.w();
                let nn = predictor.nn();
                let ww = predictor.ww();
                let nee = predictor.nee();
                let ne = predictor.ne();
                (6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub(crate) prediction: i32,
    pub(crate) max_error: i32,
    pub(crate) subpred: [i32; 4],
}

#[derive(Debug)]
pub struct PredictorState {
    width: u32,
    channel_index: u32,
    stream_index: u32,
    second_prev_row: Vec<i32>,
    prev_row: Vec<i32>,
    curr_row: Vec<i32>,
    prev_channels: Vec<PrevChannelState>,
    self_correcting: Option<SelfCorrectingPredictor>,
    x: u32,
    y: u32,
    prev_prop9: i32,
}

#[derive(Debug)]
struct PrevChannelState {
    width: u32,
    prev_row: Vec<i32>,
    curr_row: Vec<i32>,
}

impl PrevChannelState {
    fn new(width: u32) -> Self {
        Self {
            width,
            prev_row: Vec::new(),
            curr_row: Vec::new(),
        }
    }

    fn w(&self) -> i32 {
        if let Some(x) = self.curr_row.len().checked_sub(1) {
            self.curr_row[x]
        } else {
            0
        }
    }

    fn n(&self) -> i32 {
        let x = self.curr_row.len();
        match (x.checked_sub(1), self.prev_row.is_empty()) {
            (_, false) => self.prev_row[x],
            (Some(x), true) => self.curr_row[x],
            (None, true) => 0,
        }
    }

    fn nw(&self) -> i32 {
        let x = self.curr_row.len();
        match (x.checked_sub(1), self.prev_row.is_empty()) {
            (Some(x), false) => self.prev_row[x],
            (Some(x), true) => self.curr_row[x],
            (None, _) => 0,
        }
    }

    fn record(&mut self, sample: i32) {
        self.curr_row.push(sample);
        if self.curr_row.len() >= self.width as usize {
            self.prev_row = std::mem::take(&mut self.curr_row);
        }
    }
}

impl PredictorState {
    pub fn new(width: u32, channel_index: u32, stream_index: u32, prev_channels: usize, wp_header: Option<WpHeader>) -> Self {
        let self_correcting = wp_header.map(|wp_header| SelfCorrectingPredictor::new(width, wp_header));
        Self {
            width,
            channel_index,
            stream_index,
            second_prev_row: Vec::new(),
            prev_row: Vec::new(),
            curr_row: Vec::new(),
            prev_channels: (0..prev_channels).map(|_| PrevChannelState::new(width)).collect(),
            self_correcting,
            x: 0,
            y: 0,
            prev_prop9: 0,
        }
    }

    pub fn properties<'p, 's>(&'p mut self, prev_channel_samples: &'s [i32]) -> Properties<'p, 's> {
        let prediction = self.sc_predict();
        Properties::new(self, prev_channel_samples, prediction)
    }

    fn sc_predict(&self) -> Option<PredictionResult> {
        let SelfCorrectingPredictor {
            width,
            true_err_prev_row,
            true_err_curr_row,
            subpred_err_prev_row,
            subpred_err_curr_row,
            wp,
        } = self.self_correcting.as_ref()?;
        let width = *width;

        let n3 = self.n() << 3;
        let nw3 = self.nw() << 3;
        let ne3 = self.ne() << 3;
        let w3 = self.w() << 3;
        let nn3 = self.nn() << 3;

        let x = self.x as usize;

        let true_err_n = true_err_prev_row.get(x).copied().unwrap_or(0);
        let true_err_w = x.checked_sub(1)
            .and_then(|x| true_err_curr_row.get(x))
            .copied()
            .unwrap_or(0);
        let true_err_nw = x.checked_sub(1)
            .and_then(|x| true_err_prev_row.get(x))
            .copied()
            .unwrap_or(true_err_n);
        let true_err_ne = true_err_prev_row
            .get(x + 1)
            .copied()
            .unwrap_or(true_err_n);

        let subpred_err_n = subpred_err_prev_row.get(x).copied().unwrap_or_default();
        let subpred_err_w = x.checked_sub(1)
            .and_then(|x| subpred_err_curr_row.get(x))
            .copied()
            .unwrap_or_default();
        let subpred_err_ww = x.checked_sub(2)
            .and_then(|x| subpred_err_curr_row.get(x))
            .copied()
            .unwrap_or_default();
        let subpred_err_nw = x.checked_sub(1)
            .and_then(|x| subpred_err_prev_row.get(x))
            .copied()
            .unwrap_or(subpred_err_n);
        let subpred_err_ne = subpred_err_prev_row
            .get(x + 1)
            .copied()
            .unwrap_or(subpred_err_n);

        let subpred = [
            w3 + ne3 - n3,
            n3 - (((true_err_w + true_err_n + true_err_ne) * wp.wp_p1 as i32) >> 5),
            w3 - (((true_err_w + true_err_n + true_err_nw) * wp.wp_p2 as i32) >> 5),
            n3 - ((true_err_nw * wp.wp_p3a as i32 +
                    true_err_n * wp.wp_p3b as i32 +
                    true_err_ne * wp.wp_p3c as i32 +
                    (nn3 - n3) * wp.wp_p3d as i32 +
                    (nw3 - w3) * wp.wp_p3e as i32) >> 5),
        ];

        let mut subpred_err_sum = [0u32; 4];
        for (i, sum) in subpred_err_sum.iter_mut().enumerate() {
            *sum = subpred_err_n[i] + subpred_err_w[i] + subpred_err_ww[i] + subpred_err_nw[i] + subpred_err_ne[i];
            if x + 1 == width as usize {
                *sum += subpred_err_w[i];
            }
        }

        let wp_wn = [
            wp.wp_w0,
            wp.wp_w1,
            wp.wp_w2,
            wp.wp_w3,
        ];
        let mut weight = [0u32; 4];
        for ((w, err_sum), maxweight) in weight.iter_mut().zip(subpred_err_sum).zip(wp_wn) {
            let shift = (err_sum + 2).next_power_of_two().trailing_zeros().saturating_sub(6);
            *w = 4 + ((maxweight * ((1 << 24) / ((err_sum >> shift) + 1))) >> shift);
        }

        let sum_weights: u32 = weight.iter().copied().sum();
        let log_weight = (sum_weights + 1).next_power_of_two().trailing_zeros();
        for w in &mut weight {
            *w >>= log_weight.saturating_sub(5);
        }
        let sum_weights: u32 = weight.iter().copied().sum();
        let sum_weights = sum_weights as i32;
        let mut s = (sum_weights >> 1) - 1;
        for (subpred, weight) in subpred.into_iter().zip(weight) {
            s += subpred * weight as i32;
        }
        let mut prediction = ((s as i64 * ((1 << 24) / sum_weights) as i64) >> 24) as i32;
        if (true_err_n ^ true_err_w) | (true_err_n ^ true_err_nw) <= 0 {
            let min = n3.min(w3).min(ne3);
            let max = n3.max(w3).max(ne3);
            prediction = prediction.clamp(min, max);
        }

        let true_errors = [true_err_n, true_err_nw, true_err_ne];
        let mut max_error = true_err_w;
        for err in true_errors {
            if err.abs() > max_error.abs() {
                max_error = err;
            }
        }

        Some(PredictionResult { prediction, max_error, subpred })
    }
}

impl PredictorState {
    fn w(&self) -> i32 {
        let x = self.curr_row.len();
        match (x.checked_sub(1), self.prev_row.is_empty()) {
            (Some(x), _) => self.curr_row[x],
            (None, false) => self.prev_row[x],
            (None, true) => 0,
        }
    }

    fn n(&self) -> i32 {
        let x = self.curr_row.len();
        match (x.checked_sub(1), self.prev_row.is_empty()) {
            (_, false) => self.prev_row[x],
            (Some(x), true) => self.curr_row[x],
            (None, true) => 0,
        }
    }

    fn nn(&self) -> i32 {
        if self.second_prev_row.is_empty() {
            self.n()
        } else {
            self.second_prev_row[self.curr_row.len()]
        }
    }

    fn nw(&self) -> i32 {
        let x = self.curr_row.len();
        match (x.checked_sub(1), self.prev_row.is_empty()) {
            (Some(x), false) => self.prev_row[x],
            (Some(x), true) => self.curr_row[x],
            (None, false) => self.prev_row[x],
            (None, true) => 0,
        }
    }

    fn ne(&self) -> i32 {
        let x = self.curr_row.len();
        match self.prev_row.get(x + 1) {
            None => self.n(),
            Some(val) => *val,
        }
    }

    fn nee(&self) -> i32 {
        let x = self.curr_row.len();
        match self.prev_row.get(x + 2) {
            None => self.ne(),
            Some(val) => *val,
        }
    }

    fn ww(&self) -> i32 {
        let x = self.curr_row.len();
        if let Some(x) = x.checked_sub(2) {
            self.curr_row[x]
        } else {
            self.w()
        }
    }
}

#[derive(Debug)]
struct SelfCorrectingPredictor {
    width: u32,
    true_err_prev_row: Vec<i32>,
    true_err_curr_row: Vec<i32>,
    subpred_err_prev_row: Vec<[u32; 4]>,
    subpred_err_curr_row: Vec<[u32; 4]>,
    wp: WpHeader,
}

impl SelfCorrectingPredictor {
    fn new(width: u32, wp_header: WpHeader) -> Self {
        Self {
            width,
            true_err_prev_row: Vec::new(),
            true_err_curr_row: Vec::new(),
            subpred_err_prev_row: Vec::new(),
            subpred_err_curr_row: Vec::new(),
            wp: wp_header,
        }
    }

    fn record(&mut self, pred: PredictionResult, sample: i32) {
        let true_err = pred.prediction - (sample << 3);
        let mut subpred_err = [0u32; 4];
        for (err, subpred) in subpred_err.iter_mut().zip(pred.subpred) {
            *err = ((subpred - (sample << 3)).unsigned_abs() + 3) >> 3;
        }
        self.true_err_curr_row.push(true_err);
        self.subpred_err_curr_row.push(subpred_err);
        if self.true_err_curr_row.len() >= self.width as usize {
            self.true_err_prev_row = std::mem::take(&mut self.true_err_curr_row);
            self.subpred_err_prev_row = std::mem::take(&mut self.subpred_err_curr_row);
        }
    }
}

#[derive(Debug)]
pub struct Properties<'p, 's> {
    predictor: &'p mut PredictorState,
    prev_channel_samples: &'s [i32],
    sc_prediction: Option<PredictionResult>,
    cache: Vec<Cell<Option<i32>>>,
}

impl<'p, 's> Properties<'p, 's> {
    fn new(predictor: &'p mut PredictorState, prev_channel_samples: &'s [i32], sc_prediction: Option<PredictionResult>) -> Self {
        let property_count = 16 + prev_channel_samples.len() * 4;
        let cache = vec![Cell::new(None); property_count];
        Self {
            predictor,
            prev_channel_samples,
            sc_prediction,
            cache,
        }
    }
}

impl Properties<'_, '_> {
    fn prediction(&self) -> Option<i32> {
        self.sc_prediction.as_ref().map(|pred| pred.prediction)
    }

    pub fn get(&self, property: usize) -> Result<i32> {
        let Some(cache_cell) = self.cache.get(property) else {
            return Err(Error::PropertyNotFound {
                num_properties: self.cache.len(),
                property_ref: property,
            });
        };

        if let Some(val) = cache_cell.get() {
            return Ok(val);
        }

        let val = if let Some(property) = property.checked_sub(16) {
            let rev_channel_idx = property / 4;
            let prop_idx = property % 4;
            let prev_channel_idx = self.prev_channel_samples.len() - rev_channel_idx - 1;
            let c = self.prev_channel_samples[prev_channel_idx];
            let prev_channel = &self.predictor.prev_channels[prev_channel_idx];
            if prop_idx == 0 {
                c.abs()
            } else if prop_idx == 1 {
                c
            } else {
                let w = prev_channel.w();
                let n = prev_channel.n();
                let nw = prev_channel.nw();
                let g = (w + n - nw).clamp(w.min(n), w.max(n));
                if prop_idx == 2 {
                    (c - g).abs()
                } else {
                    c - g
                }
            }
        } else {
            let pred = &*self.predictor;
            match property {
                0 => pred.channel_index as i32,
                1 => pred.stream_index as i32,
                2 => pred.y as i32,
                3 => pred.x as i32,
                4 => pred.n().abs(),
                5 => pred.w().abs(),
                6 => pred.n(),
                7 => pred.w(),
                8 => pred.w() - pred.prev_prop9,
                9 => pred.n() + pred.w() - pred.nw(),
                10 => pred.w() - pred.nw(),
                11 => pred.nw() - pred.n(),
                12 => pred.n() - pred.ne(),
                13 => pred.n() - pred.nn(),
                14 => pred.w() - pred.ww(),
                15 => {
                    if let Some(prediction) = &self.sc_prediction {
                        prediction.max_error
                    } else {
                        0
                    }
                },
                _ => unreachable!(),
            }
        };
        cache_cell.set(Some(val));
        Ok(val)
    }

    pub fn record(self, sample: i32) {
        let prop9 = self.get(9).unwrap();
        let pred = self.predictor;
        if let (Some(sc), Some(pred)) = (&mut pred.self_correcting, self.sc_prediction) {
            sc.record(pred, sample);
        }

        pred.curr_row.push(sample);
        if pred.curr_row.len() >= pred.width as usize {
            pred.second_prev_row = std::mem::take(&mut pred.prev_row);
            pred.prev_row = std::mem::take(&mut pred.curr_row);
            pred.prev_prop9 = 0;
        } else {
            pred.prev_prop9 = prop9;
        }

        for (ch, sample) in pred.prev_channels.iter_mut().zip(self.prev_channel_samples) {
            ch.record(*sample);
        }
    }
}
