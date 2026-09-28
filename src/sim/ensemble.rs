//! Deterministic spin ensemble construction.
//!
//! Position and off-resonance coordinates use separate Halton bases. The low
//! discrepancy sequence covers the sample without a random generator, stored
//! seed or visible clustering, and every prefix is itself a useful ensemble.
//!
//! Receiver weights sum to one. Signal values are therefore normalized by
//! construction and do not change scale when the requested spin count changes.

use crate::sim::model::{SimulationConfig, SpinParams};

pub fn build_ensemble(config: &SimulationConfig) -> Vec<SpinParams> {
    let count = config.spin_count;
    let weight = 1.0 / count as f32;
    let mut spins = Vec::with_capacity(count);

    for index in 0..count {
        let sample = index as u32 + 1;
        let position = [
            (halton(sample, 2) - 0.5) * config.sample_extent_m[0],
            (halton(sample, 3) - 0.5) * config.sample_extent_m[1],
            (halton(sample, 5) - 0.5) * config.sample_extent_m[2],
        ];
        let offset_hz = config.center_offset_hz
            + (halton(sample, 7) - 0.5) * config.offset_span_hz;
        let t1_s = config.t1_s
            * (1.0 + (halton(sample, 11) * 2.0 - 1.0) * config.t1_spread);
        let t2_s = config.t2_s
            * (1.0 + (halton(sample, 13) * 2.0 - 1.0) * config.t2_spread);

        spins.push(SpinParams::new(
            position,
            1.0,
            t1_s,
            t2_s,
            offset_hz,
            weight,
        ));
    }
    spins
}

fn halton(mut index: u32, base: u32) -> f32 {
    let inverse = 1.0 / base as f32;
    let mut fraction = inverse;
    let mut value = 0.0f32;

    while index > 0 {
        value += (index % base) as f32 * fraction;
        index /= base;
        fraction *= inverse;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensemble_has_normalized_receiver_weight() {
        let config = SimulationConfig {
            spin_count: 1_003,
            ..SimulationConfig::default()
        };
        let spins = build_ensemble(&config);
        let weight: f32 = spins.iter().map(SpinParams::weight).sum();

        assert_eq!(spins.len(), config.spin_count);
        assert!((weight - 1.0).abs() < 2.0e-5);
    }

    #[test]
    fn ensemble_is_deterministic() {
        let config = SimulationConfig {
            spin_count: 32,
            ..SimulationConfig::default()
        };
        assert_eq!(build_ensemble(&config), build_ensemble(&config));
    }

    #[test]
    fn relaxation_spreads_produce_distinct_records() {
        let config = SimulationConfig {
            spin_count: 64,
            t1_spread: 0.25,
            t2_spread: 0.40,
            ..SimulationConfig::default()
        };
        let spins = build_ensemble(&config);

        let first = spins[0].rates_offset_weight;
        assert!(spins.iter().any(|spin| {
            spin.rates_offset_weight[0] != first[0]
                && spin.rates_offset_weight[1] != first[1]
        }));
        assert!(spins.iter().all(|spin| {
            spin.inv_t1().is_finite()
                && spin.inv_t2().is_finite()
                && spin.inv_t1() > 0.0
                && spin.inv_t2() > 0.0
        }));
    }
}