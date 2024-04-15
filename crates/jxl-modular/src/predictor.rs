use jxl_bitstream::define_bundle;
use jxl_grid::CutGrid;

use crate::Sample;

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
            _ => {
                return Err(jxl_bitstream::Error::InvalidEnum {
                    name: "MaProperty",
                    value,
                })
            }
        })
    }
}

impl Predictor {
    pub(super) fn predict<S: Sample, const EDGE: bool>(self, properties: &Properties<S>) -> i32 {
        use Predictor::*;
        let predictor = &*properties.predictor;

        match self {
            Zero => 0,
            West => predictor.w,
            North => predictor.n,
            AvgWestAndNorth => ((predictor.w as i64 + predictor.n as i64) / 2) as i32,
            Select => {
                let n = predictor.n;
                let w = predictor.w;
                let nw = predictor.nw;
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w
                } else {
                    n
                }
            }
            Gradient => {
                let n = predictor.n as i64;
                let w = predictor.w as i64;
                let nw = predictor.nw as i64;
                (n + w - nw).clamp(w.min(n), w.max(n)) as i32
            }
            SelfCorrecting => {
                let prediction = properties
                    .prediction()
                    .expect("predict_non_sc called with SelfCorrecting predictor");
                ((prediction + 3) >> 3) as i32
            }
            NorthEast => predictor.ne::<EDGE>(),
            NorthWest => predictor.nw,
            WestWest => predictor.ww::<EDGE>(),
            AvgWestAndNorthWest => ((predictor.w as i64 + predictor.nw as i64) / 2) as i32,
            AvgNorthAndNorthWest => ((predictor.n as i64 + predictor.nw as i64) / 2) as i32,
            AvgNorthAndNorthEast => {
                ((predictor.n as i64 + predictor.ne::<EDGE>() as i64) / 2) as i32
            }
            AvgAll => {
                let n = predictor.n as i64;
                let w = predictor.w as i64;
                let nn = predictor.nn::<EDGE>() as i64;
                let ww = predictor.ww::<EDGE>() as i64;
                let nee = predictor.nee::<EDGE>() as i64;
                let ne = predictor.ne::<EDGE>() as i64;
                ((6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16) as i32
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub(crate) prediction: i64,
    pub(crate) max_error: i32,
    pub(crate) subpred: [i64; 4],
}

#[derive(Debug)]
pub struct PredictorState<'prev, 'a, S: Sample> {
    width: u32,
    prev_row: Vec<i32>,
    curr_row: Vec<i32>,
    prev_channels_rev: Vec<&'a CutGrid<'prev, S>>,
    self_correcting: Option<SelfCorrectingPredictor>,
    y: u32,
    x: u32,
    w: i32,
    n: i32,
    nw: i32,
    prev_grad: i32,
}

const DIV_LOOKUP: [u32; 65] = compute_div_lookup();

const fn compute_div_lookup() -> [u32; 65] {
    let mut out = [0u32; 65];
    let mut i = 1usize;
    while i <= 64 {
        out[i] = ((1 << 24) / i) as u32;
        i += 1;
    }
    out
}

impl<'prev, 'a, S: Sample> PredictorState<'prev, 'a, S> {
    pub fn new() -> Self {
        Self {
            width: 0,
            prev_row: Vec::new(),
            curr_row: Vec::new(),
            prev_channels_rev: Vec::new(),
            self_correcting: None,
            y: 0,
            x: 0,
            w: 0,
            n: 0,
            nw: 0,
            prev_grad: 0,
        }
    }

    pub fn reset(
        &mut self,
        width: u32,
        prev_channels_rev: &[&'a CutGrid<'prev, S>],
        wp_header: Option<&WpHeader>,
    ) {
        self.self_correcting =
            wp_header.map(|wp_header| SelfCorrectingPredictor::new(width, wp_header.clone()));
        self.prev_row.clear();
        self.curr_row.clear();
        if let Some(additional) = (width as usize).checked_sub(self.prev_row.capacity()) {
            self.prev_row.reserve(additional);
        }
        if let Some(additional) = (width as usize).checked_sub(self.curr_row.capacity()) {
            self.curr_row.reserve(additional);
        }
        self.prev_channels_rev = prev_channels_rev.to_vec();
        self.width = width;
        self.y = 0;
        self.x = 0;
        self.w = 0;
        self.n = 0;
        self.nw = 0;
        self.prev_grad = 0;
    }

    #[inline]
    pub fn properties<'p, const EDGE: bool>(&'p mut self) -> Properties<'p, 'prev, 'a, S> {
        let prediction = self
            .self_correcting
            .as_ref()
            .map(|sc_pred| self.run_sc_pred::<EDGE>(sc_pred));
        Properties::new::<EDGE>(self, prediction)
    }

    fn run_sc_pred<const EDGE: bool>(&self, sc_pred: &SelfCorrectingPredictor) -> PredictionResult {
        sc_pred.predict(
            self.n,
            self.nw,
            self.ne::<EDGE>(),
            self.w,
            self.nn::<EDGE>(),
        )
    }
}

impl<'prev, 'a, S: Sample> PredictorState<'prev, 'a, S> {
    #[inline]
    fn nn<const EDGE: bool>(&self) -> i32 {
        if EDGE {
            self.curr_row
                .get(self.x as usize)
                .copied()
                .unwrap_or(self.n)
        } else {
            self.curr_row[self.x as usize]
        }
    }

    #[inline]
    fn ne<const EDGE: bool>(&self) -> i32 {
        let x = self.x;
        if EDGE && (self.prev_row.is_empty() || x + 1 >= self.width) {
            self.n
        } else {
            self.prev_row[x as usize + 1]
        }
    }

    #[inline]
    fn nee<const EDGE: bool>(&self) -> i32 {
        let x = self.x;
        if EDGE && (self.prev_row.is_empty() || x + 2 >= self.width) {
            self.ne::<true>()
        } else {
            self.prev_row[x as usize + 2]
        }
    }

    #[inline]
    fn ww<const EDGE: bool>(&self) -> i32 {
        let x = self.x;
        if EDGE {
            if let Some(x) = x.checked_sub(2) {
                self.curr_row[x as usize]
            } else {
                self.w
            }
        } else {
            self.curr_row[(x - 2) as usize]
        }
    }
}

#[derive(Debug)]
struct SelfCorrectingPredictor {
    width: u32,
    x: u32,
    y: u32,
    true_err_row: Vec<i32>,
    subpred_err_row: Vec<[u32; 4]>,
    wp: WpHeader,
    true_err_w: i32,
    true_err_nw: i32,
    true_err_n: i32,
    true_err_ne: i32,
    subpred_err_nw_ww: [u32; 4],
    subpred_err_n_w: [u32; 4],
    subpred_err_ne: [u32; 4],
}

impl SelfCorrectingPredictor {
    fn new(width: u32, wp_header: WpHeader) -> Self {
        Self {
            width,
            x: 0,
            y: 0,
            true_err_row: vec![0i32; width as usize],
            subpred_err_row: vec![[0u32; 4]; width as usize],
            wp: wp_header,
            true_err_w: 0,
            true_err_nw: 0,
            true_err_n: 0,
            true_err_ne: 0,
            subpred_err_nw_ww: [0; 4],
            subpred_err_n_w: [0; 4],
            subpred_err_ne: [0; 4],
        }
    }

    fn predict(&self, n: i32, nw: i32, ne: i32, w: i32, nn: i32) -> PredictionResult {
        let SelfCorrectingPredictor {
            ref wp,
            true_err_w,
            true_err_nw,
            true_err_n,
            true_err_ne,
            subpred_err_nw_ww,
            subpred_err_n_w,
            subpred_err_ne,
            ..
        } = *self;
        let true_err_w = true_err_w as i64;
        let true_err_nw = true_err_nw as i64;
        let true_err_n = true_err_n as i64;
        let true_err_ne = true_err_ne as i64;

        let n3 = (n as i64) << 3;
        let nw3 = (nw as i64) << 3;
        let ne3 = (ne as i64) << 3;
        let w3 = (w as i64) << 3;
        let nn3 = (nn as i64) << 3;

        let subpred = [
            w3 + ne3 - n3,
            n3 - (((true_err_w + true_err_n + true_err_ne) * wp.wp_p1 as i64) >> 5),
            w3 - (((true_err_w + true_err_n + true_err_nw) * wp.wp_p2 as i64) >> 5),
            n3 - ((true_err_nw * wp.wp_p3a as i64
                + true_err_n * wp.wp_p3b as i64
                + true_err_ne * wp.wp_p3c as i64
                + (nn3 - n3) * wp.wp_p3d as i64
                + (nw3 - w3) * wp.wp_p3e as i64)
                >> 5),
        ];

        let mut subpred_err_sum = [0u32; 4];
        for (i, sum) in subpred_err_sum.iter_mut().enumerate() {
            *sum = subpred_err_nw_ww[i]
                .wrapping_add(subpred_err_n_w[i])
                .wrapping_add(subpred_err_ne[i]);
        }

        let wp_wn = [wp.wp_w0, wp.wp_w1, wp.wp_w2, wp.wp_w3];
        let mut weight = [0u32; 4];
        for ((w, err_sum), maxweight) in weight.iter_mut().zip(subpred_err_sum).zip(wp_wn) {
            let shift = ((err_sum as u64 + 1) >> 5).checked_ilog2().unwrap_or(0);
            *w = 4 + ((maxweight * DIV_LOOKUP[(err_sum >> shift) as usize + 1]) >> shift);
        }

        let sum_weights: u32 = weight.iter().copied().sum();
        let log_weight = (sum_weights as u64 >> 4).ilog2();
        for w in &mut weight {
            *w >>= log_weight;
        }
        let sum_weights: u32 = weight.iter().copied().sum();
        let mut s = (sum_weights as i64 >> 1) - 1;
        for (subpred, weight) in subpred.into_iter().zip(weight) {
            s += subpred * weight as i64;
        }
        let mut prediction = (s * DIV_LOOKUP[sum_weights as usize] as i64) >> 24;
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

        PredictionResult {
            prediction,
            max_error: max_error as i32,
            subpred,
        }
    }

    fn record(&mut self, pred: PredictionResult, sample: i32) {
        let sample = sample as i64;
        let true_err = pred.prediction - (sample << 3);
        let mut subpred_err = [0u32; 4];
        for (err, subpred) in subpred_err.iter_mut().zip(pred.subpred) {
            *err = ((subpred.abs_diff(sample << 3) + 3) >> 3) as u32;
        }

        self.true_err_row[self.x as usize] = true_err as i32;
        self.subpred_err_row[self.x as usize] = subpred_err;
        self.x += 1;

        if self.x >= self.width {
            self.y += 1;
            self.x = 0;

            self.true_err_w = 0;
            self.true_err_n = self.true_err_row[0];
            self.true_err_nw = self.true_err_n;
            self.subpred_err_n_w = self.subpred_err_row[0];
            self.subpred_err_nw_ww = self.subpred_err_n_w;
            if self.width <= 1 {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else {
                self.true_err_ne = self.true_err_row[1];
                self.subpred_err_ne = self.subpred_err_row[1];
            }
        } else {
            self.true_err_w = true_err as i32;
            self.true_err_nw = self.true_err_n;
            self.true_err_n = self.true_err_ne;
            self.subpred_err_nw_ww = self.subpred_err_n_w;
            self.subpred_err_n_w = self.subpred_err_ne;
            for (w0, w1) in self.subpred_err_n_w.iter_mut().zip(subpred_err) {
                *w0 = w0.wrapping_add(w1);
            }

            if self.x + 1 >= self.width {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else if self.y != 0 {
                let x = self.x as usize;
                self.true_err_ne = self.true_err_row[x + 1];
                self.subpred_err_ne = self.subpred_err_row[x + 1];
            }
        }
    }
}

#[derive(Debug)]
pub struct Properties<'p, 'prev, 'a, S: Sample> {
    predictor: &'p mut PredictorState<'prev, 'a, S>,
    sc_prediction: Option<PredictionResult>,
    prop_cache: [i32; 16],
}

impl<'p, 'prev, 'a, S: Sample> Properties<'p, 'prev, 'a, S> {
    fn new<const EDGE: bool>(
        pred: &'p mut PredictorState<'prev, 'a, S>,
        sc_prediction: Option<PredictionResult>,
    ) -> Self {
        let w_nw = pred.w.wrapping_sub(pred.nw);
        let prop_cache = [
            0,
            0,
            pred.y as i32,
            pred.x as i32,
            pred.n.unsigned_abs() as i32,
            pred.w.unsigned_abs() as i32,
            pred.n,
            pred.w,
            pred.w.wrapping_sub(pred.prev_grad),
            w_nw.wrapping_add(pred.n),
            w_nw,
            pred.nw.wrapping_sub(pred.n),
            pred.n.wrapping_sub(pred.ne::<EDGE>()),
            pred.n.wrapping_sub(pred.nn::<EDGE>()),
            pred.w.wrapping_sub(pred.ww::<EDGE>()),
            sc_prediction
                .as_ref()
                .map(|pred| pred.max_error)
                .unwrap_or(0),
        ];

        Self {
            predictor: pred,
            sc_prediction,
            prop_cache,
        }
    }
}

impl<S: Sample> Properties<'_, '_, '_, S> {
    #[inline]
    fn prediction(&self) -> Option<i64> {
        self.sc_prediction.as_ref().map(|pred| pred.prediction)
    }

    #[inline]
    fn get_extra(&self, prop_extra: usize) -> i32 {
        let prev_channel_idx = prop_extra / 4;
        let prop_idx = prop_extra % 4;

        let Some(prev_channel) = self.predictor.prev_channels_rev.get(prev_channel_idx) else {
            return 0;
        };
        let x = self.predictor.x as usize;
        let y = self.predictor.y as usize;

        let c = prev_channel.get(x, y).to_i32();
        if prop_idx == 0 {
            c.abs()
        } else if prop_idx == 1 {
            c
        } else {
            let g = match (x, y) {
                (0, 0) => 0,
                (0, y) => prev_channel.get(0, y - 1).to_i32(),
                (x, 0) => prev_channel.get(x - 1, 0).to_i32(),
                (x, y) => {
                    let rn = prev_channel.get_row(y - 1);
                    let nw = rn[x - 1];
                    let n = rn[x];
                    let w = prev_channel.get(x - 1, y);
                    S::grad_clamped(n, w, nw).to_i32()
                }
            };
            if prop_idx == 2 {
                c.abs_diff(g) as i32
            } else {
                c.wrapping_sub(g)
            }
        }
    }

    #[inline(always)]
    pub fn get(&self, property: usize) -> i32 {
        if let Some(property) = property.checked_sub(16) {
            self.get_extra(property)
        } else {
            self.prop_cache[property]
        }
    }

    pub fn record(self, sample: i32) {
        let pred = self.predictor;
        if let (Some(sc), Some(pred)) = (&mut pred.self_correcting, self.sc_prediction) {
            sc.record(pred, sample);
        }

        if let Some(out) = pred.curr_row.get_mut(pred.x as usize) {
            *out = sample;
        } else {
            pred.curr_row.push(sample);
        }
        pred.x += 1;

        if pred.x >= pred.width {
            pred.y += 1;
            pred.x = 0;

            std::mem::swap(&mut pred.prev_row, &mut pred.curr_row);
            pred.prev_grad = 0;

            let n = pred.prev_row[0];
            pred.n = n;
            pred.w = n;
            pred.nw = n;
        } else {
            pred.prev_grad = self.prop_cache[9];

            pred.w = sample;
            if pred.prev_row.is_empty() {
                pred.nw = sample;
                pred.n = sample;
            } else {
                pred.nw = pred.n;
                pred.n = pred.prev_row[pred.x as usize];
            }
        }
    }
}
