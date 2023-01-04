use jxl_bitstream::{define_bundle, read_bits, Bitstream};

use crate::Grid;

use super::ModularChannelInfo;

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

    fn try_from(value: u32) -> Result<Self, Self::Error> {
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
    pub(super) fn predict(self, grid: &Grid<i32>, x: i32, y: i32, prediction_result: &PredictionResult) -> i32 {
        if self == Predictor::SelfCorrecting {
            (prediction_result.prediction + 3) >> 3
        } else {
            self.predict_non_sc(grid, x, y)
        }
    }

    pub(super) fn predict_non_sc(self, grid: &Grid<i32>, x: i32, y: i32) -> i32 {
        use Predictor::*;
        let anchor = grid.anchor(x, y);

        match self {
            Zero => 0,
            West => anchor.w(),
            North => anchor.n(),
            AvgWestAndNorth => (anchor.w() + anchor.n()) / 2,
            Select => {
                let n = anchor.n();
                let w = anchor.w();
                let nw = anchor.nw();
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            },
            Gradient => {
                let n = anchor.n();
                let w = anchor.w();
                let nw = anchor.nw();
                (w + n - nw).clamp(w.min(n), w.max(n))
            },
            SelfCorrecting => panic!("predict_non_sc called with SelfCorrecting predictor"),
            NorthEast => anchor.ne(),
            NorthWest => anchor.nw(),
            WestWest => anchor.ww(),
            AvgWestAndNorthWest => (anchor.w() + anchor.nw()) / 2,
            AvgNorthAndNorthWest => (anchor.n() + anchor.nw()) / 2,
            AvgNorthAndNorthEast => (anchor.n() + anchor.ne()) / 2,
            AvgAll => {
                let n = anchor.n();
                let w = anchor.w();
                let nn = anchor.nn();
                let ww = anchor.ww();
                let nee = anchor.nee();
                let ne = anchor.ne();
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
pub struct SelfCorrectingPredictor {
    width: u32,
    true_err_prev_row: Vec<i32>,
    true_err_curr_row: Vec<i32>,
    subpred_err_prev_row: Vec<[u32; 4]>,
    subpred_err_curr_row: Vec<[u32; 4]>,
    wp: WpHeader,
}

impl SelfCorrectingPredictor {
    pub fn new(width: u32, wp_header: WpHeader) -> Self {
        Self {
            width,
            true_err_prev_row: Vec::new(),
            true_err_curr_row: Vec::new(),
            subpred_err_prev_row: Vec::new(),
            subpred_err_curr_row: Vec::new(),
            wp: wp_header,
        }
    }

    pub fn predict(&self, grid: &Grid<i32>, x: i32, y: i32) -> PredictionResult {
        let Self {
            width,
            true_err_prev_row,
            true_err_curr_row,
            subpred_err_prev_row,
            subpred_err_curr_row,
            wp,
        } = self;
        let width = *width;

        let anchor = grid.anchor(x, y);
        let n3 = anchor.n() << 3;
        let nw3 = anchor.nw() << 3;
        let ne3 = anchor.ne() << 3;
        let w3 = anchor.w() << 3;
        let nn3 = anchor.nn() << 3;

        let x = x as usize;

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

        PredictionResult { prediction, max_error, subpred }
    }

    pub fn record_error(&mut self, prediction: PredictionResult, true_value: i32) {
        let true_err = prediction.prediction - (true_value << 3);
        let mut subpred_err = [0u32; 4];
        for (err, subpred) in subpred_err.iter_mut().zip(prediction.subpred) {
            *err = ((subpred - (true_value << 3)).unsigned_abs() + 3) >> 3;
        }
        self.true_err_curr_row.push(true_err);
        self.subpred_err_curr_row.push(subpred_err);
        if self.true_err_curr_row.len() >= self.width as usize {
            self.true_err_prev_row = std::mem::take(&mut self.true_err_curr_row);
            self.subpred_err_prev_row = std::mem::take(&mut self.subpred_err_curr_row);
        }
    }
}
