//! Pulse sequence representation and timeline compilation.
//!
//! A SequenceProgram is an editable list of RF, gradient and ADC events on an
//! absolute time axis. Events may overlap. RF and gradient values add while ADC
//! events mark observation intervals.
//!
//! RF events state their total flip angle rather than a field amplitude. During
//! compilation the selected waveform is sampled, normalized by its signed area
//! and converted into piecewise constant B1 fields. Rectangular pulses use one
//! waveform sample. Gaussian and windowed sinc pulses use the event sample
//! count, then every resulting field interval is divided further when required
//! by the RF integration limit.
//!
//! The compiled timeline is cut at field boundaries, observation times and RF
//! integration boundaries. Every backend therefore consumes the same sequence
//! of constant field steps and records samples at the same absolute times.

use std::f32::consts::{PI, TAU};

use crate::sim::model::{SequenceKind, SimulationConfig};

const MAX_RF_STEP_S: f32 = 10.0e-6;
const TIME_EPSILON_S: f32 = 1.0e-8;
const MIN_EVENT_DURATION_S: f32 = 1.0e-6;

pub const MAX_SEQUENCE_EVENTS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceEventKind {
    Rf,
    Gradient,
    Adc,
}

impl SequenceEventKind {
    pub const LABELS: [&'static str; 3] = ["RF", "Gradient", "ADC"];

    pub const fn index(self) -> usize {
        match self {
            SequenceEventKind::Rf => 0,
            SequenceEventKind::Gradient => 1,
            SequenceEventKind::Adc => 2,
        }
    }

    pub const fn from_index(index: usize) -> SequenceEventKind {
        match index {
            1 => SequenceEventKind::Gradient,
            2 => SequenceEventKind::Adc,
            _ => SequenceEventKind::Rf,
        }
    }

    pub const fn label(self) -> &'static str {
        Self::LABELS[self.index()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfShape {
    Rectangular,
    Gaussian,
    Sinc,
}

impl RfShape {
    pub const LABELS: [&'static str; 3] = ["Rectangular", "Gaussian", "Sinc"];

    pub const fn index(self) -> usize {
        match self {
            RfShape::Rectangular => 0,
            RfShape::Gaussian => 1,
            RfShape::Sinc => 2,
        }
    }

    pub const fn from_index(index: usize) -> RfShape {
        match index {
            1 => RfShape::Gaussian,
            2 => RfShape::Sinc,
            _ => RfShape::Rectangular,
        }
    }
}

/// One editable event.
///
/// Fields not used by the selected kind retain their values. Switching an event
/// from RF to gradient and back therefore does not discard its RF settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequenceEvent {
    pub kind: SequenceEventKind,
    pub start_s: f32,
    pub duration_s: f32,
    pub rf_flip_rad: f32,
    pub rf_phase_rad: f32,
    pub rf_shape: RfShape,
    pub rf_samples: u32,
    pub gradient_t_m: [f32; 3],
}

impl SequenceEvent {
    pub fn rf(
        center_s: f32,
        duration_s: f32,
        flip_rad: f32,
        phase_rad: f32,
        shape: RfShape,
        samples: u32,
    ) -> SequenceEvent {
        SequenceEvent {
            kind: SequenceEventKind::Rf,
            start_s: center_s - duration_s * 0.5,
            duration_s,
            rf_flip_rad: flip_rad,
            rf_phase_rad: phase_rad,
            rf_shape: shape,
            rf_samples: samples,
            gradient_t_m: [0.0; 3],
        }
    }

    pub fn gradient(
        start_s: f32,
        duration_s: f32,
        gradient_t_m: [f32; 3],
    ) -> SequenceEvent {
        SequenceEvent {
            kind: SequenceEventKind::Gradient,
            start_s,
            duration_s,
            rf_flip_rad: PI * 0.5,
            rf_phase_rad: 0.0,
            rf_shape: RfShape::Rectangular,
            rf_samples: 1,
            gradient_t_m,
        }
    }

    pub fn adc(start_s: f32, duration_s: f32) -> SequenceEvent {
        SequenceEvent {
            kind: SequenceEventKind::Adc,
            start_s,
            duration_s,
            rf_flip_rad: PI * 0.5,
            rf_phase_rad: 0.0,
            rf_shape: RfShape::Rectangular,
            rf_samples: 1,
            gradient_t_m: [0.0; 3],
        }
    }

    pub fn center_s(&self) -> f32 {
        self.start_s + self.duration_s * 0.5
    }

    pub fn converted(mut self, kind: SequenceEventKind) -> SequenceEvent {
        self.kind = kind;
        self
    }

    pub fn sanitize(&mut self, sequence_duration_s: f32) {
        let duration = finite_or(sequence_duration_s, 1.0e-3).max(MIN_EVENT_DURATION_S);
        self.start_s = finite_or(self.start_s, 0.0)
            .clamp(0.0, (duration - MIN_EVENT_DURATION_S).max(0.0));
        self.duration_s = finite_or(self.duration_s, MIN_EVENT_DURATION_S)
            .clamp(MIN_EVENT_DURATION_S, duration - self.start_s);

        self.rf_flip_rad = finite_or(self.rf_flip_rad, 0.0).clamp(0.0, TAU);
        self.rf_phase_rad = finite_or(self.rf_phase_rad, 0.0).rem_euclid(TAU);
        self.rf_samples = self.rf_samples.clamp(1, 256);

        for gradient in &mut self.gradient_t_m {
            *gradient = finite_or(*gradient, 0.0).clamp(-0.1, 0.1);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequenceProgram {
    pub duration_s: f32,
    pub echo_time_s: f32,
    pub events: Vec<SequenceEvent>,
}

impl SequenceProgram {
    pub fn new(duration_s: f32, echo_time_s: f32) -> SequenceProgram {
        let mut program = SequenceProgram {
            duration_s,
            echo_time_s,
            events: Vec::with_capacity(8),
        };
        program.sanitize();
        program
    }

    pub fn preset(config: &SimulationConfig) -> SequenceProgram {
        match config.sequence {
            SequenceKind::SpinEcho => spin_echo(config),
            SequenceKind::GradientEcho => gradient_echo(config),
            SequenceKind::Custom => SequenceProgram::new(
                config.te_s * 1.35 + config.rf_duration_s,
                config.te_s,
            ),
        }
    }

    pub fn sanitize(&mut self) {
        self.duration_s = finite_or(self.duration_s, 1.0e-3).max(1.0e-3);
        self.echo_time_s =
            finite_or(self.echo_time_s, 0.0).clamp(0.0, self.duration_s);

        if self.events.len() > MAX_SEQUENCE_EVENTS {
            self.events.truncate(MAX_SEQUENCE_EVENTS);
        }
        for event in &mut self.events {
            event.sanitize(self.duration_s);
        }
    }

    pub fn excitation_center_s(&self) -> f32 {
        self.events
            .iter()
            .filter(|event| event.kind == SequenceEventKind::Rf)
            .map(SequenceEvent::center_s)
            .min_by(f32::total_cmp)
            .unwrap_or(0.0)
    }
}

/// Constant RF and gradient fields over one absolute time interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldEvent {
    pub start_s: f32,
    pub end_s: f32,
    pub b1_t: [f32; 2],
    pub gradient_t_m: [f32; 3],
}

impl FieldEvent {
    pub fn duration_s(&self) -> f32 {
        self.end_s - self.start_s
    }
}

/// Inclusive acquisition interval on the sequence time axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdcBlock {
    pub start_s: f32,
    pub end_s: f32,
}

impl AdcBlock {
    pub fn duration_s(&self) -> f32 {
        self.end_s - self.start_s
    }

    pub fn contains(&self, time_s: f32) -> bool {
        time_s >= self.start_s - TIME_EPSILON_S
            && time_s <= self.end_s + TIME_EPSILON_S
    }
}

/// One backend execution interval after all overlapping events are combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldStep {
    pub duration_s: f32,
    pub b1_t: [f32; 2],
    pub gradient_t_m: [f32; 3],
    pub observe_after: bool,
    pub acquired: bool,
}

/// Backend-ready timeline and the field events used for visualization.
pub struct CompiledSequence {
    pub duration_s: f32,
    pub excitation_center_s: f32,
    pub echo_time_s: f32,
    pub events: Vec<FieldEvent>,
    pub adc_blocks: Vec<AdcBlock>,
    pub(crate) steps: Vec<FieldStep>,
}

impl CompiledSequence {
    pub fn build(config: &SimulationConfig) -> CompiledSequence {
        let program = SequenceProgram::preset(config);
        Self::compile(config, &program)
    }

    pub fn compile(
        config: &SimulationConfig,
        program: &SequenceProgram,
    ) -> CompiledSequence {
        let mut program = program.clone();
        program.sanitize();

        let mut fields = Vec::with_capacity(program.events.len() * 8);
        let mut adc_blocks = Vec::with_capacity(4);

        for event in &program.events {
            match event.kind {
                SequenceEventKind::Rf => append_rf(config, event, &mut fields),
                SequenceEventKind::Gradient => fields.push(FieldEvent {
                    start_s: event.start_s,
                    end_s: event.start_s + event.duration_s,
                    b1_t: [0.0; 2],
                    gradient_t_m: event.gradient_t_m,
                }),
                SequenceEventKind::Adc => adc_blocks.push(AdcBlock {
                    start_s: event.start_s,
                    end_s: event.start_s + event.duration_s,
                }),
            }
        }

        fields.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
        adc_blocks.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));

        let mut observations = Vec::with_capacity(config.observation_count + 8);
        for index in 0..config.observation_count {
            observations.push(
                program.duration_s * index as f32
                    / (config.observation_count - 1).max(1) as f32,
            );
        }
        observations.push(program.echo_time_s);
        for adc in &adc_blocks {
            observations.push(adc.start_s);
            observations.push(adc.end_s);
        }
        sort_unique(&mut observations);

        let mut cuts = Vec::with_capacity(observations.len() + fields.len() * 8);
        cuts.extend_from_slice(&observations);
        cuts.push(0.0);
        cuts.push(program.duration_s);

        for event in &fields {
            cuts.push(event.start_s);
            cuts.push(event.end_s);

            if event.b1_t[0] != 0.0 || event.b1_t[1] != 0.0 {
                let parts =
                    (event.duration_s() / MAX_RF_STEP_S).ceil().max(1.0) as usize;
                for part in 1..parts {
                    cuts.push(
                        event.start_s
                            + event.duration_s() * part as f32 / parts as f32,
                    );
                }
            }
        }
        sort_unique(&mut cuts);

        let mut steps = Vec::with_capacity(cuts.len().saturating_sub(1));
        for pair in cuts.windows(2) {
            let start_s = pair[0];
            let end_s = pair[1];
            let duration_s = end_s - start_s;
            if duration_s <= TIME_EPSILON_S {
                continue;
            }

            let midpoint = (start_s + end_s) * 0.5;
            let mut b1_t = [0.0f32; 2];
            let mut gradient_t_m = [0.0f32; 3];

            for event in &fields {
                if midpoint >= event.start_s && midpoint < event.end_s {
                    b1_t[0] += event.b1_t[0];
                    b1_t[1] += event.b1_t[1];
                    for axis in 0..3 {
                        gradient_t_m[axis] += event.gradient_t_m[axis];
                    }
                }
            }

            steps.push(FieldStep {
                duration_s,
                b1_t,
                gradient_t_m,
                observe_after: contains_time(&observations, end_s),
                acquired: adc_blocks.iter().any(|adc| adc.contains(end_s)),
            });
        }

        CompiledSequence {
            duration_s: program.duration_s,
            excitation_center_s: program.excitation_center_s(),
            echo_time_s: program.echo_time_s,
            events: fields,
            adc_blocks,
            steps,
        }
    }

    pub fn steps(&self) -> &[FieldStep] {
        &self.steps
    }
}

fn append_rf(
    config: &SimulationConfig,
    event: &SequenceEvent,
    output: &mut Vec<FieldEvent>,
) {
    if event.duration_s <= 0.0 || event.rf_flip_rad == 0.0 {
        return;
    }

    let samples = match event.rf_shape {
        RfShape::Rectangular => 1usize,
        _ => event.rf_samples.clamp(2, 256) as usize,
    };
    let sample_s = event.duration_s / samples as f32;

    let mut area = 0.0f32;
    for index in 0..samples {
        let position = (index as f32 + 0.5) / samples as f32;
        area += rf_weight(event.rf_shape, position);
    }
    if area.abs() < 1.0e-6 {
        area = samples as f32;
    }

    let scale =
        event.rf_flip_rad / (config.gamma_rad_s_t * sample_s * area);
    let phase = event.rf_phase_rad;
    let axis = [phase.cos(), phase.sin()];

    for index in 0..samples {
        let start_s = event.start_s + index as f32 * sample_s;
        let weight = rf_weight(event.rf_shape, (index as f32 + 0.5) / samples as f32);
        let amplitude = scale * weight;

        output.push(FieldEvent {
            start_s,
            end_s: start_s + sample_s,
            b1_t: [axis[0] * amplitude, axis[1] * amplitude],
            gradient_t_m: [0.0; 3],
        });
    }
}

fn rf_weight(shape: RfShape, position: f32) -> f32 {
    match shape {
        RfShape::Rectangular => 1.0,
        RfShape::Gaussian => {
            let x = (position - 0.5) * 6.0;
            (-0.5 * x * x).exp()
        }
        RfShape::Sinc => {
            let x = (position - 0.5) * 6.0;
            let sinc = if x.abs() < 1.0e-6 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let window = 0.54 + 0.46 * (PI * x / 3.0).cos();
            sinc * window
        }
    }
}

fn spin_echo(config: &SimulationConfig) -> SequenceProgram {
    let te = config.te_s;
    let pulse_s = config
        .rf_duration_s
        .clamp(20.0e-6, (te * 0.20).max(20.0e-6));
    let adc_s = config
        .adc_duration_s
        .clamp(0.2e-3, (te * 0.80).max(0.2e-3));
    let excitation_center = 1.5e-3 + pulse_s * 0.5;
    let echo_time = excitation_center + te;
    let duration =
        (echo_time + (te * 0.35).max(adc_s * 0.6).max(6.0e-3))
            .max(echo_time + adc_s * 0.5 + 1.0e-3);
    let refocus_center = excitation_center + te * 0.5;
    let gradient_s = (te * 0.08).clamp(0.4e-3, 3.0e-3);
    let clearance = pulse_s * 0.75 + 0.3e-3;
    let first_end = refocus_center - pulse_s * 0.5 - clearance;
    let first_start =
        (first_end - gradient_s).max(excitation_center + pulse_s * 0.5 + 0.2e-3);
    let actual_gradient_s = (first_end - first_start).max(MIN_EVENT_DURATION_S);
    let second_start = refocus_center + pulse_s * 0.5 + clearance;
    let gradient = config.gradient_amplitude_t_m;

    let mut program = SequenceProgram::new(duration, echo_time);
    program.events.push(SequenceEvent::rf(
        excitation_center,
        pulse_s,
        config.excitation_flip_rad,
        config.rf_phase_rad,
        RfShape::Rectangular,
        1,
    ));
    program.events.push(SequenceEvent::rf(
        refocus_center,
        pulse_s,
        config.refocus_flip_rad,
        config.rf_phase_rad,
        RfShape::Rectangular,
        1,
    ));
    program.events.push(SequenceEvent::gradient(
        first_start,
        actual_gradient_s,
        [gradient, gradient * 0.25, 0.0],
    ));
    program.events.push(SequenceEvent::gradient(
        second_start,
        actual_gradient_s,
        [gradient, gradient * 0.25, 0.0],
    ));
    program.events.push(SequenceEvent::adc(
        echo_time - adc_s * 0.5,
        adc_s,
    ));
    program.sanitize();
    program
}

fn gradient_echo(config: &SimulationConfig) -> SequenceProgram {
    let te = config.te_s;
    let pulse_s = config
        .rf_duration_s
        .clamp(20.0e-6, (te * 0.20).max(20.0e-6));
    let adc_s = config
        .adc_duration_s
        .clamp(0.2e-3, (te * 0.80).max(0.2e-3));
    let excitation_center = 1.5e-3 + pulse_s * 0.5;
    let echo_time = excitation_center + te;
    let duration =
        (echo_time + (te * 0.35).max(adc_s * 0.6).max(6.0e-3))
            .max(echo_time + adc_s * 0.5 + 1.0e-3);

    let prephase_start = excitation_center + pulse_s * 0.5 + 0.3e-3;
    let prephase_s = (te * 0.18).clamp(0.8e-3, 8.0e-3);
    let read_start = excitation_center + te * 0.42;
    let balance_s = (echo_time - read_start).max(0.2e-3);
    let prephase_gradient = -config.gradient_amplitude_t_m;
    let read_gradient = -prephase_gradient * prephase_s / balance_s;
    let phase_s = prephase_s * 0.45;
    let phase_gradient = config.gradient_amplitude_t_m * 0.40;

    let mut program = SequenceProgram::new(duration, echo_time);
    program.events.push(SequenceEvent::rf(
        excitation_center,
        pulse_s,
        config.excitation_flip_rad,
        config.rf_phase_rad,
        RfShape::Rectangular,
        1,
    ));
    program.events.push(SequenceEvent::gradient(
        prephase_start,
        prephase_s,
        [prephase_gradient, 0.0, 0.0],
    ));
    program.events.push(SequenceEvent::gradient(
        read_start,
        echo_time + adc_s * 0.5 - read_start,
        [read_gradient, 0.0, 0.0],
    ));
    program.events.push(SequenceEvent::gradient(
        prephase_start,
        phase_s,
        [0.0, phase_gradient, 0.0],
    ));
    program.events.push(SequenceEvent::gradient(
        prephase_start + phase_s + 0.2e-3,
        phase_s,
        [0.0, -phase_gradient, 0.0],
    ));
    program.events.push(SequenceEvent::adc(
        echo_time - adc_s * 0.5,
        adc_s,
    ));
    program.sanitize();
    program
}

fn sort_unique(values: &mut Vec<f32>) {
    values.retain(|value| value.is_finite());
    values.sort_by(f32::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= TIME_EPSILON_S);
}

fn contains_time(values: &[f32], wanted: f32) -> bool {
    values
        .iter()
        .any(|value| (*value - wanted).abs() <= TIME_EPSILON_S)
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_timelines_cover_their_duration() {
        for kind in [SequenceKind::SpinEcho, SequenceKind::GradientEcho] {
            let config = SimulationConfig {
                sequence: kind,
                ..SimulationConfig::default()
            };
            let sequence = CompiledSequence::build(&config);
            let total: f32 =
                sequence.steps.iter().map(|step| step.duration_s).sum();

            assert!((total - sequence.duration_s).abs() < 2.0e-6);
            assert!(sequence.steps.iter().all(|step| step.duration_s > 0.0));
            assert!(sequence.echo_time_s > sequence.excitation_center_s);
        }
    }

    #[test]
    fn rf_steps_respect_the_integration_limit() {
        let config = SimulationConfig {
            rf_duration_s: 1.2e-3,
            ..SimulationConfig::default()
        };
        let sequence = CompiledSequence::build(&config);

        for step in &sequence.steps {
            if step.b1_t[0] != 0.0 || step.b1_t[1] != 0.0 {
                assert!(step.duration_s <= MAX_RF_STEP_S + TIME_EPSILON_S);
            }
        }
    }

    #[test]
    fn shaped_rf_preserves_the_requested_flip_area() {
        let config = SimulationConfig::default();

        for shape in [RfShape::Rectangular, RfShape::Gaussian, RfShape::Sinc] {
            let mut program = SequenceProgram::new(0.01, 0.005);
            program.events.push(SequenceEvent::rf(
                0.003,
                1.0e-3,
                PI * 0.5,
                0.0,
                shape,
                64,
            ));

            let compiled = CompiledSequence::compile(&config, &program);
            let angle: f32 = compiled
                .steps()
                .iter()
                .map(|step| config.gamma_rad_s_t * step.b1_t[0] * step.duration_s)
                .sum();

            assert!((angle - PI * 0.5).abs() < 2.0e-4, "{shape:?}: {angle}");
        }
    }

    #[test]
    fn custom_events_may_overlap() {
        let config = SimulationConfig {
            sequence: SequenceKind::Custom,
            ..SimulationConfig::default()
        };
        let mut program = SequenceProgram::new(0.02, 0.012);
        program.events.push(SequenceEvent::gradient(
            0.004,
            0.006,
            [0.001, 0.0, 0.0],
        ));
        program.events.push(SequenceEvent::gradient(
            0.007,
            0.006,
            [0.0, -0.002, 0.0],
        ));
        program.events.push(SequenceEvent::adc(0.010, 0.004));

        let compiled = CompiledSequence::compile(&config, &program);
        assert!(compiled.steps().iter().any(|step| {
            step.gradient_t_m[0] != 0.0 && step.gradient_t_m[1] != 0.0
        }));
        assert!(compiled.steps().iter().any(|step| step.acquired));
    }

    #[test]
    fn empty_sequence_keeps_equally_spaced_observations() {
        let config = SimulationConfig {
            sequence: SequenceKind::Custom,
            observation_count: 97,
            ..SimulationConfig::default()
        };
        let program = SequenceProgram::new(0.08, 0.04);
        let compiled = CompiledSequence::compile(&config, &program);

        assert!(compiled.events.is_empty());
        assert!(compiled.adc_blocks.is_empty());
        assert_eq!(
            compiled.steps().iter().filter(|step| step.observe_after).count(),
            96
        );
        assert!(compiled.steps().iter().map(|step| step.duration_s).sum::<f32>()
            - program.duration_s)
            .abs()
    }

    #[test]
    fn multiple_adc_blocks_retain_separate_acquired_ranges() {
        let config = SimulationConfig {
            sequence: SequenceKind::Custom,
            observation_count: 201,
            ..SimulationConfig::default()
        };
        let mut program = SequenceProgram::new(0.10, 0.05);
        program.events.push(SequenceEvent::adc(0.020, 0.010));
        program.events.push(SequenceEvent::adc(0.065, 0.012));

        let compiled = CompiledSequence::compile(&config, &program);
        assert_eq!(compiled.adc_blocks.len(), 2);

        let mut time = 0.0f32;
        let mut first = false;
        let mut second = false;
        for step in compiled.steps() {
            time += step.duration_s;
            if step.acquired && time < 0.04 {
                first = true;
            }
            if step.acquired && time > 0.06 {
                second = true;
            }
        }
        assert!(first && second);
    }
}