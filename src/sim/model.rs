//! Spin parameters and backend independent simulation configuration.
//!
//! Frequency offsets are angular frequencies in the rotating frame. Positions
//! are metres, fields are tesla, gradients are tesla per metre and all times
//! are seconds.
//!
//! The two spin structures are aligned to sixteen bytes and arranged as two
//! vec4 compatible records. The layout can be copied into storage buffers
//! without a second representation or per-field packing step.

use std::f32::consts::TAU;

/// Proton gyromagnetic ratio in radians per second per tesla.
pub const PROTON_GAMMA_RAD: f32 = 2.675_221_9e8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceKind {
    SpinEcho,
    GradientEcho,
    Custom,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinParams {
    /// Position in metres and equilibrium longitudinal magnetization.
    pub position_m_m0: [f32; 4],
    /// Inverse T1, inverse T2, angular frequency offset and receiver weight.
    pub rates_offset_weight: [f32; 4],
}

impl SpinParams {
    pub fn new(
        position_m: [f32; 3],
        m0: f32,
        t1_s: f32,
        t2_s: f32,
        offset_hz: f32,
        weight: f32,
    ) -> SpinParams {
        SpinParams {
            position_m_m0: [position_m[0], position_m[1], position_m[2], m0],
            rates_offset_weight: [
                1.0 / t1_s.max(1.0e-9),
                1.0 / t2_s.max(1.0e-9),
                offset_hz * TAU,
                weight,
            ],
        }
    }

    pub fn position_m(&self) -> [f32; 3] {
        [
            self.position_m_m0[0],
            self.position_m_m0[1],
            self.position_m_m0[2],
        ]
    }

    pub fn m0(&self) -> f32 {
        self.position_m_m0[3]
    }

    pub fn inv_t1(&self) -> f32 {
        self.rates_offset_weight[0]
    }

    pub fn inv_t2(&self) -> f32 {
        self.rates_offset_weight[1]
    }

    pub fn offset_rad_s(&self) -> f32 {
        self.rates_offset_weight[2]
    }

    pub fn weight(&self) -> f32 {
        self.rates_offset_weight[3]
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinState {
    /// Mx, My, Mz and a reserved word.
    pub magnetization: [f32; 4],
}

impl SpinState {
    pub const fn new(mx: f32, my: f32, mz: f32) -> SpinState {
        SpinState { magnetization: [mx, my, mz, 0.0] }
    }

    pub fn equilibrium(params: &SpinParams) -> SpinState {
        SpinState::new(0.0, 0.0, params.m0())
    }

    pub fn mx(&self) -> f32 {
        self.magnetization[0]
    }

    pub fn my(&self) -> f32 {
        self.magnetization[1]
    }

    pub fn mz(&self) -> f32 {
        self.magnetization[2]
    }
}

/// Physical ensemble and sampling parameters shared by every backend.
///
/// The structure contains no editable event list. SequenceProgram carries that
/// list separately so presets and custom timelines can use the same physical
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationConfig {
    pub sequence: SequenceKind,
    pub spin_count: usize,
    pub observation_count: usize,
    pub t1_s: f32,
    pub t2_s: f32,
    /// Half width of the relative T1 distribution.
    pub t1_spread: f32,
    /// Half width of the relative T2 distribution.
    pub t2_spread: f32,
    pub center_offset_hz: f32,
    /// Full width of the uniform frequency distribution.
    pub offset_span_hz: f32,
    pub te_s: f32,
    pub excitation_flip_rad: f32,
    pub refocus_flip_rad: f32,
    pub rf_phase_rad: f32,
    pub rf_duration_s: f32,
    pub gradient_amplitude_t_m: f32,
    pub adc_duration_s: f32,
    pub sample_extent_m: [f32; 3],
    pub gamma_rad_s_t: f32,
}

impl SimulationConfig {
    pub fn sanitized(mut self) -> SimulationConfig {
        self.spin_count = self.spin_count.clamp(8, 1_048_576);
        self.observation_count = self.observation_count.clamp(32, 2_048);
        self.t1_s = self.t1_s.max(1.0e-6);
        self.t2_s = self.t2_s.max(1.0e-6);
        self.t1_spread = self.t1_spread.clamp(0.0, 0.95);
        self.t2_spread = self.t2_spread.clamp(0.0, 0.95);
        self.offset_span_hz = self.offset_span_hz.max(0.0);
        self.te_s = self.te_s.max(2.0e-3);
        self.excitation_flip_rad = self.excitation_flip_rad.clamp(0.0, std::f32::consts::TAU);
        self.refocus_flip_rad = self.refocus_flip_rad.clamp(0.0, std::f32::consts::TAU);
        self.rf_phase_rad = self.rf_phase_rad.rem_euclid(std::f32::consts::TAU);
        self.rf_duration_s = self.rf_duration_s.max(20.0e-6);
        self.gradient_amplitude_t_m =
            self.gradient_amplitude_t_m.abs().min(0.1);
        self.adc_duration_s = self.adc_duration_s.max(0.2e-3);
        self.gamma_rad_s_t = self.gamma_rad_s_t.abs().max(1.0);

        for extent in &mut self.sample_extent_m {
            *extent = extent.abs().max(1.0e-6);
        }
        self
    }
}

impl Default for SimulationConfig {
    fn default() -> SimulationConfig {
        SimulationConfig {
            sequence: SequenceKind::SpinEcho,
            spin_count: 512,
            observation_count: 361,
            t1_s: 0.9,
            t2_s: 0.08,
            t1_spread: 0.0,
            t2_spread: 0.0,
            center_offset_hz: 18.0,
            offset_span_hz: 120.0,
            te_s: 0.06,
            excitation_flip_rad: std::f32::consts::PI * 0.5,
            refocus_flip_rad: std::f32::consts::PI,
            rf_phase_rad: 0.0,
            rf_duration_s: 0.5e-3,
            gradient_amplitude_t_m: 0.18e-3,
            adc_duration_s: 12.0e-3,
            sample_extent_m: [0.12, 0.12, 0.02],
            gamma_rad_s_t: PROTON_GAMMA_RAD,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SignalSample {
    pub time_s: f32,
    pub real: f32,
    pub imaginary: f32,
    pub magnitude: f32,
    pub acquired: bool,
}

/// Signal and visualization state produced by one complete backend run.
///
/// Visual states are a deterministic prefix of the full ensemble. Frame n
/// occupies n * visual_spin_count .. (n + 1) * visual_spin_count, matching the
/// signal sample at the same index.
pub struct SimulationTrace {
    pub config: SimulationConfig,
    pub sequence_revision: u64,
    pub samples: Vec<SignalSample>,
    pub visual_spin_count: usize,
    pub visual_states: Vec<SpinState>,
}

impl SimulationTrace {
    pub fn sample_index_at(&self, time_s: f32) -> usize {
        if self.samples.is_empty() {
            return 0;
        }

        let wanted = time_s.max(0.0);
        match self
            .samples
            .binary_search_by(|sample| sample.time_s.total_cmp(&wanted))
        {
            Ok(index) => index,
            Err(0) => 0,
            Err(index) if index >= self.samples.len() => self.samples.len() - 1,
            Err(index) => {
                let before = self.samples[index - 1].time_s;
                let after = self.samples[index].time_s;
                if wanted - before <= after - wanted {
                    index - 1
                } else {
                    index
                }
            }
        }
    }

    pub fn sample_at(&self, time_s: f32) -> SignalSample {
        if self.samples.is_empty() {
            SignalSample::default()
        } else {
            self.samples[self.sample_index_at(time_s)]
        }
    }

    pub fn states_at(&self, time_s: f32) -> &[SpinState] {
        if self.visual_spin_count == 0 || self.visual_states.is_empty() {
            return &[];
        }

        let frame = self.sample_index_at(time_s);
        let start = frame * self.visual_spin_count;
        let end = start + self.visual_spin_count;
        if end <= self.visual_states.len() {
            &self.visual_states[start..end]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_records_match_vec4_layout() {
        assert_eq!(std::mem::size_of::<SpinParams>(), 32);
        assert_eq!(std::mem::align_of::<SpinParams>(), 16);
        assert_eq!(std::mem::size_of::<SpinState>(), 16);
        assert_eq!(std::mem::align_of::<SpinState>(), 16);
    }

    #[test]
    fn frequency_input_is_stored_as_angular_frequency() {
        let spin = SpinParams::new([0.0; 3], 1.0, 1.0, 1.0, 25.0, 1.0);
        assert!((spin.offset_rad_s() - TAU * 25.0).abs() < 1.0e-4);
    }
}