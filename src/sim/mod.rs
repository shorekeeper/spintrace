//! Spin ensemble simulation.
//!
//! The public model is independent of the execution backend. Spin parameters,
//! magnetization state, compiled field steps and signal samples have no Vulkan
//! handles and can be shared by the CPU reference path, tests and the compute
//! backend.
//!
//! The CPU implementation is the numerical reference. It evaluates the same
//! piecewise constant fields the compute shader will consume and retains spin
//! snapshots at observation times for the vector and phase views.

pub mod cpu;
pub mod ensemble;
pub mod model;
pub mod sequence;

pub use cpu::{CpuPreview, RunOutput};
pub use model::{
    SignalSample, SimulationConfig, SimulationTrace, SpinParams, SpinState,
    SequenceKind, PROTON_GAMMA_RAD,
};
pub use sequence::{
    AdcBlock, CompiledSequence, FieldEvent, RfShape, SequenceEvent,
    SequenceEventKind, SequenceProgram, MAX_SEQUENCE_EVENTS,
};