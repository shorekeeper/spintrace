//! CPU Bloch integrator and retained preview output.
//!
//! Free precession under a longitudinal field is integrated analytically,
//! including T1 and T2 relaxation. A step containing transverse RF applies
//! half a relaxation step, an exact Rodrigues rotation around the complete
//! effective field, then the remaining half relaxation step.
//!
//! The sign convention is dM/dt = M cross omega. Under a positive longitudinal
//! frequency, Mx + iMy therefore accumulates phase as exp(-i omega t).
//!
//! Output snapshots are stored in one flat array. Frame n occupies
//! n * spin_count .. (n + 1) * spin_count, which avoids one allocation per
//! observation and matches the contiguous form needed by a later readback path.

use crate::sim::ensemble::build_ensemble;
use crate::sim::model::{
    SignalSample, SimulationConfig, SpinParams, SpinState,
};
use crate::sim::sequence::{CompiledSequence, FieldStep, SequenceProgram};

/// Complete CPU reference result with signal samples and spin snapshots.
pub struct RunOutput {
    config: SimulationConfig,
    sequence: CompiledSequence,
    spins: Vec<SpinParams>,
    samples: Vec<SignalSample>,
    state_frames: Vec<SpinState>,
}

impl RunOutput {
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    pub fn sequence(&self) -> &CompiledSequence {
        &self.sequence
    }

    pub fn spins(&self) -> &[SpinParams] {
        &self.spins
    }

    pub fn samples(&self) -> &[SignalSample] {
        &self.samples
    }

    pub fn spin_count(&self) -> usize {
        self.spins.len()
    }

    pub fn duration_s(&self) -> f32 {
        self.sequence.duration_s
    }

    pub fn echo_time_s(&self) -> f32 {
        self.sequence.echo_time_s
    }

    pub fn frame_index_at(&self, time_s: f32) -> usize {
        if self.samples.is_empty() {
            return 0;
        }

        let wanted = time_s.clamp(0.0, self.duration_s());
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

    pub fn states_at(&self, time_s: f32) -> &[SpinState] {
        if self.spins.is_empty() || self.samples.is_empty() {
            return &[];
        }

        let frame = self.frame_index_at(time_s);
        let start = frame * self.spins.len();
        let end = start + self.spins.len();
        &self.state_frames[start..end]
    }

    pub fn sample_at(&self, time_s: f32) -> SignalSample {
        if self.samples.is_empty() {
            return SignalSample::default();
        }
        self.samples[self.frame_index_at(time_s)]
    }
}

/// Retained CPU reference that rebuilds only when its inputs change.
pub struct CpuPreview {
    config: SimulationConfig,
    program: SequenceProgram,
    output: RunOutput,
}

impl CpuPreview {
    pub fn new(config: SimulationConfig) -> CpuPreview {
        let config = config.sanitized();
        let program = SequenceProgram::preset(&config);
        CpuPreview::with_program(config, program)
    }

    pub fn with_program(
        config: SimulationConfig,
        program: SequenceProgram,
    ) -> CpuPreview {
        let config = config.sanitized();
        let output = run_program(config, &program);
        CpuPreview { config, program, output }
    }

    /// Rebuilds only when a physical, sampling or sequence value changed.
    pub fn update_program(
        &mut self,
        config: SimulationConfig,
        program: &SequenceProgram,
    ) -> bool {
        let config = config.sanitized();
        if config == self.config && program == &self.program {
            return false;
        }

        self.output = run_program(config, program);
        self.config = config;
        self.program = program.clone();
        true
    }

    pub fn output(&self) -> &RunOutput {
        &self.output
    }
}

pub fn run(config: SimulationConfig) -> RunOutput {
    let config = config.sanitized();
    let program = SequenceProgram::preset(&config);
    run_program(config, &program)
}

pub fn run_program(
    config: SimulationConfig,
    program: &SequenceProgram,
) -> RunOutput {
    let config = config.sanitized();
    let sequence = CompiledSequence::compile(&config, program);
    let spins = build_ensemble(&config);
    let mut states: Vec<SpinState> = spins.iter().map(SpinState::equilibrium).collect();
    let mut samples = Vec::with_capacity(config.observation_count + 4);
    let mut state_frames =
        Vec::with_capacity((config.observation_count + 4) * spins.len());

    record_sample(
        0.0,
        false,
        &spins,
        &states,
        &mut samples,
        &mut state_frames,
    );

    let mut time_s = 0.0f32;
    for step in sequence.steps() {
        for (params, state) in spins.iter().zip(states.iter_mut()) {
            advance_spin(
                params,
                state,
                step,
                config.gamma_rad_s_t,
            );
        }

        time_s += step.duration_s;
        if step.observe_after {
            record_sample(
                time_s,
                step.acquired,
                &spins,
                &states,
                &mut samples,
                &mut state_frames,
            );
        }
    }

    RunOutput { config, sequence, spins, samples, state_frames }
}

pub fn advance_spin(
    params: &SpinParams,
    state: &mut SpinState,
    step: &FieldStep,
    gamma_rad_s_t: f32,
) {
    let position = params.position_m();
    let omega_z = params.offset_rad_s()
        + gamma_rad_s_t
            * (step.gradient_t_m[0] * position[0]
                + step.gradient_t_m[1] * position[1]
                + step.gradient_t_m[2] * position[2]);

    if step.b1_t[0] == 0.0 && step.b1_t[1] == 0.0 {
        advance_free(params, state, omega_z, step.duration_s);
        return;
    }

    relax(params, state, step.duration_s * 0.5);

    let omega = [
        gamma_rad_s_t * step.b1_t[0],
        gamma_rad_s_t * step.b1_t[1],
        omega_z,
    ];
    rotate(state, omega, step.duration_s);

    relax(params, state, step.duration_s * 0.5);
}

fn advance_free(
    params: &SpinParams,
    state: &mut SpinState,
    omega_z: f32,
    duration_s: f32,
) {
    let transverse = (-duration_s * params.inv_t2()).exp();
    let angle = omega_z * duration_s;
    let cosine = angle.cos();
    let sine = angle.sin();
    let mx = state.magnetization[0];
    let my = state.magnetization[1];

    state.magnetization[0] = transverse * (mx * cosine + my * sine);
    state.magnetization[1] = transverse * (my * cosine - mx * sine);

    let longitudinal = (-duration_s * params.inv_t1()).exp();
    state.magnetization[2] =
        params.m0() + (state.magnetization[2] - params.m0()) * longitudinal;
}

fn relax(params: &SpinParams, state: &mut SpinState, duration_s: f32) {
    let transverse = (-duration_s * params.inv_t2()).exp();
    let longitudinal = (-duration_s * params.inv_t1()).exp();

    state.magnetization[0] *= transverse;
    state.magnetization[1] *= transverse;
    state.magnetization[2] =
        params.m0() + (state.magnetization[2] - params.m0()) * longitudinal;
}

fn rotate(state: &mut SpinState, omega: [f32; 3], duration_s: f32) {
    let magnitude =
        (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    if magnitude <= 1.0e-12 {
        return;
    }

    let axis = [
        omega[0] / magnitude,
        omega[1] / magnitude,
        omega[2] / magnitude,
    ];
    let angle = magnitude * duration_s;
    let cosine = angle.cos();
    let sine = angle.sin();
    let one_minus_cosine = 1.0 - cosine;

    let value = [
        state.magnetization[0],
        state.magnetization[1],
        state.magnetization[2],
    ];
    let cross = [
        value[1] * axis[2] - value[2] * axis[1],
        value[2] * axis[0] - value[0] * axis[2],
        value[0] * axis[1] - value[1] * axis[0],
    ];
    let projection =
        value[0] * axis[0] + value[1] * axis[1] + value[2] * axis[2];

    for component in 0..3 {
        state.magnetization[component] = value[component] * cosine
            + cross[component] * sine
            + axis[component] * projection * one_minus_cosine;
    }
}


fn record_sample(
    time_s: f32,
    acquired: bool,
    spins: &[SpinParams],
    states: &[SpinState],
    samples: &mut Vec<SignalSample>,
    state_frames: &mut Vec<SpinState>,
) {
    let mut real = 0.0f32;
    let mut imaginary = 0.0f32;

    for (params, state) in spins.iter().zip(states) {
        real += params.weight() * state.mx();
        imaginary += params.weight() * state.my();
    }

    samples.push(SignalSample {
        time_s,
        real,
        imaginary,
        magnitude: (real * real + imaginary * imaginary).sqrt(),
        acquired,
    });
    state_frames.extend_from_slice(states);
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{PI, TAU};

    use super::*;
    use crate::sim::model::{SequenceKind, PROTON_GAMMA_RAD};

    fn free_step(duration_s: f32) -> FieldStep {
        FieldStep {
            duration_s,
            b1_t: [0.0; 2],
            gradient_t_m: [0.0; 3],
            observe_after: false,
            acquired: false,
        }
    }

    #[test]
    fn free_precession_matches_phase_and_t2() {
        let params = SpinParams::new([0.0; 3], 1.0, 100.0, 0.08, 25.0, 1.0);
        let mut state = SpinState::new(1.0, 0.0, 0.0);
        advance_spin(&params, &mut state, &free_step(0.01), PROTON_GAMMA_RAD);

        let amplitude = (-0.01f32 / 0.08).exp();
        let phase = -TAU * 25.0 * 0.01;
        assert!((state.mx() - amplitude * phase.cos()).abs() < 1.0e-5);
        assert!((state.my() - amplitude * phase.sin()).abs() < 1.0e-5);
    }

    #[test]
    fn longitudinal_recovery_matches_the_exponential() {
        let params = SpinParams::new([0.0; 3], 1.0, 0.7, 100.0, 0.0, 1.0);
        let mut state = SpinState::new(0.0, 0.0, 0.0);
        advance_spin(&params, &mut state, &free_step(0.4), PROTON_GAMMA_RAD);

        let expected = 1.0 - (-0.4f32 / 0.7).exp();
        assert!((state.mz() - expected).abs() < 1.0e-6);
    }

    #[test]
    fn resonant_rectangular_pulse_produces_ninety_degrees() {
        let duration_s = 1.0e-3;
        let params = SpinParams::new([0.0; 3], 1.0, 1.0e6, 1.0e6, 0.0, 1.0);
        let mut state = SpinState::equilibrium(&params);
        let step = FieldStep {
            duration_s,
            b1_t: [PI * 0.5 / (PROTON_GAMMA_RAD * duration_s), 0.0],
            gradient_t_m: [0.0; 3],
            observe_after: false,
            acquired: false,
        };

        advance_spin(&params, &mut state, &step, PROTON_GAMMA_RAD);
        assert!(state.mx().abs() < 1.0e-5);
        assert!((state.my() - 1.0).abs() < 1.0e-5);
        assert!(state.mz().abs() < 1.0e-5);
    }

    #[test]
    fn spin_echo_refocuses_static_offsets() {
        let config = SimulationConfig {
            sequence: SequenceKind::SpinEcho,
            spin_count: 511,
            observation_count: 601,
            t1_s: 100.0,
            t2_s: 0.5,
            center_offset_hz: 0.0,
            offset_span_hz: 240.0,
            te_s: 0.06,
            ..SimulationConfig::default()
        };
        let output = run(config);
        let echo = output.sample_at(output.echo_time_s()).magnitude;
        let before = output.sample_at(output.echo_time_s() - 0.008).magnitude;
        let after = output.sample_at(output.echo_time_s() + 0.008).magnitude;

        assert!(echo > 0.65, "echo {}", echo);
        assert!(echo > before * 2.5, "echo {} before {}", echo, before);
        assert!(echo > after * 2.5, "echo {} after {}", echo, after);
    }

    #[test]
    fn gradient_moment_forms_an_echo() {
        let config = SimulationConfig {
            sequence: SequenceKind::GradientEcho,
            spin_count: 509,
            observation_count: 601,
            t1_s: 100.0,
            t2_s: 0.5,
            center_offset_hz: 0.0,
            offset_span_hz: 0.0,
            te_s: 0.06,
            ..SimulationConfig::default()
        };
        let output = run(config);
        let echo = output.sample_at(output.echo_time_s()).magnitude;
        let before = output.sample_at(output.echo_time_s() - 0.012).magnitude;

        assert!(echo > 0.75, "echo {}", echo);
        assert!(echo > before * 2.0, "echo {} before {}", echo, before);
    }
}