use jxl_bitstream::define_bundle;

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
    pub(super) fn predict(self, properties: &Properties<'_, '_>) -> i64 {
        use Predictor::*;
        let predictor = &*properties.predictor;

        match self {
            Zero => 0,
            West => predictor.w as i64,
            North => predictor.n as i64,
            AvgWestAndNorth => (predictor.w as i64 + predictor.n as i64) / 2,
            Select => {
                let n = predictor.n;
                let w = predictor.w;
                let nw = predictor.nw;
                if n.abs_diff(nw) < w.abs_diff(nw) {
                    w as i64
                } else {
                    n as i64
                }
            },
            Gradient => {
                let n = predictor.n as i64;
                let w = predictor.w as i64;
                let nw = predictor.nw as i64;
                (n + w - nw).clamp(w.min(n), w.max(n))
            },
            SelfCorrecting => {
                let prediction = properties
                    .prediction()
                    .expect("predict_non_sc called with SelfCorrecting predictor");
                (prediction + 3) >> 3
            },
            NorthEast => predictor.ne() as i64,
            NorthWest => predictor.nw as i64,
            WestWest => predictor.ww() as i64,
            AvgWestAndNorthWest => (predictor.w as i64 + predictor.nw as i64) / 2,
            AvgNorthAndNorthWest => (predictor.n as i64 + predictor.nw as i64) / 2,
            AvgNorthAndNorthEast => (predictor.n as i64 + predictor.ne() as i64) / 2,
            AvgAll => {
                let n = predictor.n as i64;
                let w = predictor.w as i64;
                let nn = predictor.nn() as i64;
                let ww = predictor.ww() as i64;
                let nee = predictor.nee() as i64;
                let ne = predictor.ne() as i64;
                (6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16
            },
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
pub struct PredictorState {
    width: u32,
    channel_index: u32,
    stream_index: u32,
    second_prev_row: Vec<i32>,
    prev_row: Vec<i32>,
    curr_row: Vec<i32>,
    prev_channels_rev: Vec<PrevChannelState>,
    self_correcting: Option<SelfCorrectingPredictor>,
    y: u32,
    w: i32,
    n: i32,
    nw: i32,
    prev_grad: i32,
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
            prev_row: Vec::with_capacity(width as usize),
            curr_row: Vec::with_capacity(width as usize),
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
            std::mem::swap(&mut self.prev_row, &mut self.curr_row);
            self.curr_row.clear();
        }
    }
}

impl PredictorState {
    const DIV_LOOKUP: [u32; 65] = Self::compute_div_lookup();

    const fn compute_div_lookup() -> [u32; 65] {
        let mut out = [0u32; 65];
        let mut i = 1usize;
        while i <= 64 {
            out[i] = ((1 << 24) / i) as u32;
            i += 1;
        }
        out
    }

    pub fn new(width: u32, channel_index: u32, stream_index: u32, prev_channels: usize, wp_header: Option<&WpHeader>) -> Self {
        let self_correcting = wp_header.map(|wp_header| SelfCorrectingPredictor::new(width, wp_header.clone()));
        Self {
            width,
            channel_index,
            stream_index,
            second_prev_row: Vec::with_capacity(width as usize),
            prev_row: Vec::with_capacity(width as usize),
            curr_row: Vec::with_capacity(width as usize),
            prev_channels_rev: (0..prev_channels).map(|_| PrevChannelState::new(width)).collect(),
            self_correcting,
            y: 0,
            w: 0,
            n: 0,
            nw: 0,
            prev_grad: 0,
        }
    }

    pub fn properties<'p, 's>(&'p mut self, prev_channel_samples_rev: &'s [i32]) -> Properties<'p, 's> {
        let prediction = self.sc_predict();
        Properties::new(self, prev_channel_samples_rev, prediction)
    }

    fn sc_predict(&self) -> Option<PredictionResult> {
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
        } = *self.self_correcting.as_ref()?;
        let true_err_w = true_err_w as i64;
        let true_err_nw = true_err_nw as i64;
        let true_err_n = true_err_n as i64;
        let true_err_ne = true_err_ne as i64;

        let n3 = (self.n as i64) << 3;
        let nw3 = (self.nw as i64) << 3;
        let ne3 = (self.ne() as i64) << 3;
        let w3 = (self.w as i64) << 3;
        let nn3 = (self.nn() as i64) << 3;

        let subpred = [
            w3 + ne3 - n3,
            n3 - (((true_err_w + true_err_n + true_err_ne) * wp.wp_p1 as i64) >> 5),
            w3 - (((true_err_w + true_err_n + true_err_nw) * wp.wp_p2 as i64) >> 5),
            n3 - ((true_err_nw * wp.wp_p3a as i64 +
                    true_err_n * wp.wp_p3b as i64 +
                    true_err_ne * wp.wp_p3c as i64 +
                    (nn3 - n3) * wp.wp_p3d as i64 +
                    (nw3 - w3) * wp.wp_p3e as i64) >> 5),
        ];

        let mut subpred_err_sum = [0u32; 4];
        for (i, sum) in subpred_err_sum.iter_mut().enumerate() {
            *sum = subpred_err_nw_ww[i].wrapping_add(subpred_err_n_w[i]).wrapping_add(subpred_err_ne[i]);
        }

        let wp_wn = [
            wp.wp_w0,
            wp.wp_w1,
            wp.wp_w2,
            wp.wp_w3,
        ];
        let mut weight = [0u32; 4];
        for ((w, err_sum), maxweight) in weight.iter_mut().zip(subpred_err_sum).zip(wp_wn) {
            let shift = floor_log2(err_sum as u64 + 1).saturating_sub(5);
            *w = 4 + ((maxweight * Self::DIV_LOOKUP[(err_sum >> shift) as usize + 1]) >> shift);
        }

        let sum_weights: u32 = weight.iter().copied().sum();
        let log_weight = floor_log2(sum_weights as u64) - 4;
        for w in &mut weight {
            *w >>= log_weight;
        }
        let sum_weights: u32 = weight.iter().copied().sum();
        let mut s = (sum_weights as i64 >> 1) - 1;
        for (subpred, weight) in subpred.into_iter().zip(weight) {
            s += subpred * weight as i64;
        }
        let mut prediction = (s * Self::DIV_LOOKUP[sum_weights as usize] as i64) >> 24;
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

        Some(PredictionResult { prediction, max_error: max_error as i32, subpred })
    }
}

impl PredictorState {
    fn nn(&self) -> i32 {
        if self.second_prev_row.is_empty() {
            self.n
        } else {
            self.second_prev_row[self.curr_row.len()]
        }
    }

    fn ne(&self) -> i32 {
        let x = self.curr_row.len();
        if self.prev_row.is_empty() || x + 1 >= self.width as usize {
            self.n
        } else {
            self.prev_row[x + 1]
        }
    }

    fn nee(&self) -> i32 {
        let x = self.curr_row.len();
        if self.prev_row.is_empty() || x + 2 >= self.width as usize {
            self.ne()
        } else {
            self.prev_row[x + 2]
        }
    }

    fn ww(&self) -> i32 {
        let x = self.curr_row.len();
        if let Some(x) = x.checked_sub(2) {
            self.curr_row[x]
        } else {
            self.w
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
            true_err_prev_row: Vec::with_capacity(width as usize),
            true_err_curr_row: Vec::with_capacity(width as usize),
            subpred_err_prev_row: Vec::with_capacity(width as usize),
            subpred_err_curr_row: Vec::with_capacity(width as usize),
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

    fn record(&mut self, pred: PredictionResult, sample: i32) {
        let sample = sample as i64;
        let true_err = pred.prediction - (sample << 3);
        let mut subpred_err = [0u32; 4];
        for (err, subpred) in subpred_err.iter_mut().zip(pred.subpred) {
            *err = ((subpred.abs_diff(sample << 3) + 3) >> 3) as u32;
        }

        self.true_err_curr_row.push(true_err as i32);
        self.subpred_err_curr_row.push(subpred_err);

        let x = self.true_err_curr_row.len();
        if x >= self.width as usize {
            std::mem::swap(&mut self.true_err_prev_row, &mut self.true_err_curr_row);
            std::mem::swap(&mut self.subpred_err_prev_row, &mut self.subpred_err_curr_row);
            self.true_err_curr_row.clear();
            self.subpred_err_curr_row.clear();

            self.true_err_w = 0;
            self.true_err_n = self.true_err_prev_row[0];
            self.true_err_nw = self.true_err_n;
            self.subpred_err_n_w = self.subpred_err_prev_row[0];
            self.subpred_err_nw_ww = self.subpred_err_n_w;
            if self.width <= 1 {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else {
                self.true_err_ne = self.true_err_prev_row[1];
                self.subpred_err_ne = self.subpred_err_prev_row[1];
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

            if x + 1 >= self.width as usize {
                self.true_err_ne = self.true_err_n;
                self.subpred_err_ne = self.subpred_err_n_w;
            } else if !self.true_err_prev_row.is_empty() {
                self.true_err_ne = self.true_err_prev_row[x + 1];
                self.subpred_err_ne = self.subpred_err_prev_row[x + 1];
            }
        }
    }
}

#[derive(Debug)]
pub struct Properties<'p, 's> {
    predictor: &'p mut PredictorState,
    prev_channel_samples_rev: &'s [i32],
    sc_prediction: Option<PredictionResult>,
    prop_cache: [i32; 16],
}

impl<'p, 's> Properties<'p, 's> {
    fn new(pred: &'p mut PredictorState, prev_channel_samples_rev: &'s [i32], sc_prediction: Option<PredictionResult>) -> Self {
        let prop_cache = [
            pred.channel_index as i32,
            pred.stream_index as i32,
            pred.y as i32,
            pred.curr_row.len() as i32,
            pred.n.abs(),
            pred.w.abs(),
            pred.n,
            pred.w,
            pred.w.wrapping_sub(pred.prev_grad),
            (pred.w as i64 + pred.n as i64 - pred.nw as i64) as i32,
            pred.w.wrapping_sub(pred.nw),
            pred.nw.wrapping_sub(pred.n),
            pred.n.wrapping_sub(pred.ne()),
            pred.n.wrapping_sub(pred.nn()),
            pred.w.wrapping_sub(pred.ww()),
            if let Some(prediction) = &sc_prediction {
                prediction.max_error
            } else {
                0
            },
        ];
        Self {
            predictor: pred,
            prev_channel_samples_rev,
            sc_prediction,
            prop_cache,
        }
    }
}

impl Properties<'_, '_> {
    fn prediction(&self) -> Option<i64> {
        self.sc_prediction.as_ref().map(|pred| pred.prediction)
    }

    #[inline]
    fn get_extra(&self, prop_extra: usize) -> i32 {
        let prev_channel_idx = prop_extra / 4;
        let prop_idx = prop_extra % 4;

        let Some(c) = self.prev_channel_samples_rev.get(prev_channel_idx).copied() else { return 0; };
        let prev_channel = &self.predictor.prev_channels_rev[prev_channel_idx];
        if prop_idx == 0 {
            c.abs()
        } else if prop_idx == 1 {
            c
        } else {
            let w = prev_channel.w() as i64;
            let n = prev_channel.n() as i64;
            let nw = prev_channel.nw() as i64;
            let g = (w + n - nw).clamp(w.min(n), w.max(n)) as i32;
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

        pred.curr_row.push(sample);
        let x = pred.curr_row.len();
        if x >= pred.width as usize {
            pred.y += 1;

            std::mem::swap(&mut pred.second_prev_row, &mut pred.prev_row);
            std::mem::swap(&mut pred.prev_row, &mut pred.curr_row);
            pred.curr_row.clear();
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
                pred.n = pred.prev_row[x];
            }
        }

        for (ch, sample) in pred.prev_channels_rev.iter_mut().zip(self.prev_channel_samples_rev) {
            ch.record(*sample);
        }
    }
}

fn floor_log2(x: u64) -> u32 {
    u64::BITS - 1 - x.leading_zeros()
}
