//! Vulkan compute pipelines for Bloch integration and signal accumulation.
//!
//! The compute subsystem owns a command pool, one primary command buffer and a
//! fence. A model request records the complete sequence and submits it without
//! waiting. Polling the fence later publishes the result while the application
//! continues processing input and rendering frames.
//!
//! A second queue from the graphics family is used when the device exposes one.
//! The resources in this module are private to that queue while a request is in
//! flight, so no semaphore or queue ownership transfer is required. A device
//! with one queue retains the same interface but compute and presentation are
//! serialized by that queue.
//!
//! Parameters and state use persistently mapped host visible storage. The state
//! buffer also serves startup verification and the source of visualization
//! snapshots. Reduction scratch remains device local. Signal partials and
//! selected spin states are copied into one compact host visible readback
//! buffer after each recorded observation.
//!
//! Spin counts are padded to a power of two. Dummy records have zero M0 and
//! receiver weight, so complete workgroups can execute without shader bounds
//! branches and cannot affect signal reduction.
//!
//! Reduction stops at at most two hundred and fifty six vec4 partials. Those
//! values are added on the CPU after the fence signals. This avoids floating
//! point atomics and transfers only a few kilobytes per signal sample.
//!
//! The constructor verifies finite RF, gradients and relaxation against the CPU
//! integrator. A mismatch prevents renderer creation.

use std::f32::consts::PI;
use std::ffi::{c_char, c_void};
use std::time::Instant;

use crate::core::{Error, Result};
use crate::render::compute_shader::{self, WORKGROUP_SIZE};
use crate::render::device::Device;
use crate::render::memory::{as_bytes, Buffer, DynamicBuffer, Location};
use crate::render::vk::*;
use crate::sim::cpu::{advance_spin, run_program};
use crate::sim::ensemble::build_ensemble;
use crate::sim::{
    RfShape, SequenceEvent, SequenceKind, SignalSample, SimulationConfig,
    SimulationTrace, SpinParams, SpinState, PROTON_GAMMA_RAD,
};
use crate::sim::sequence::{CompiledSequence, FieldStep, SequenceProgram};

const PUSH_BYTES: u32 = 32;
const SIGNAL_PARTIALS: usize = 256;
const VISUAL_SPINS: usize = 512;

/// Last completed compute run and startup verification measurements.
#[derive(Debug, Clone, Copy, Default)]
pub struct ComputeDiagnostics {
    pub workgroup_size: u32,
    pub verification_spins: u32,
    pub verification_dispatches: u32,
    pub verification_max_error: f32,
    pub signal_verification_max_error: f32,
    pub model_spins: u32,
    pub model_steps: u32,
    pub signal_samples: u32,
    pub signal_dispatches: u32,
    pub partials_per_sample: u32,
    pub visual_spins: u32,
    pub run_ms: f32,
    pub running: bool,
    pub separate_queue: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct StepPush {
    field: [f32; 4],
    gradient: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SignalPartial {
    value: [f32; 4],
}

struct SpinBuffers {
    params: Buffer,
    states: Buffer,
    scratch_a: Buffer,
    scratch_b: Buffer,
    capacity: usize,
}

impl SpinBuffers {
    fn new(device: &Device, capacity: usize) -> Result<SpinBuffers> {
        let param_bytes =
            (capacity * std::mem::size_of::<SpinParams>()) as VkDeviceSize;
        let state_bytes =
            (capacity * std::mem::size_of::<SpinState>()) as VkDeviceSize;
        let scratch_bytes =
            (capacity * std::mem::size_of::<SignalPartial>()) as VkDeviceSize;

        let mut params = Buffer::new(
            device,
            param_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            Location::HostVisible,
        )?;

        let mut states = match Buffer::new(
            device,
            state_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            Location::HostVisible,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                params.destroy(device);
                return Err(error);
            }
        };

        let scratch_usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        let mut scratch_a = match Buffer::new(
            device,
            scratch_bytes,
            scratch_usage,
            Location::DeviceLocal,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                states.destroy(device);
                params.destroy(device);
                return Err(error);
            }
        };

        let scratch_b = match Buffer::new(
            device,
            scratch_bytes,
            scratch_usage,
            Location::DeviceLocal,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                scratch_a.destroy(device);
                states.destroy(device);
                params.destroy(device);
                return Err(error);
            }
        };

        Ok(SpinBuffers { params, states, scratch_a, scratch_b, capacity })
    }

    fn destroy(&mut self, device: &Device) {
        self.params.destroy(device);
        self.states.destroy(device);
        self.scratch_a.destroy(device);
        self.scratch_b.destroy(device);
    }
}

struct PendingRun {
    config: SimulationConfig,
    sequence_revision: u64,
    observations: Vec<(f32, bool)>,
    partials: usize,
    visual_spins: usize,
    signal_bytes: usize,
    dispatches: u32,
    started: Instant,
}

/// Compute pipelines, descriptors, buffers and asynchronous submission state.
///
/// One request may be in flight. A newer requested configuration remains in the
/// application model and is submitted after the active fence signals.
pub struct SpinCompute {
    descriptor_layout: VkDescriptorSetLayout,
    layout: VkPipelineLayout,
    reset_pipeline: VkPipeline,
    evolve_pipeline: VkPipeline,
    signal_pipeline: VkPipeline,
    reduce_pipeline: VkPipeline,

    descriptor_pool: VkDescriptorPool,
    main_set: VkDescriptorSet,
    reduce_ab_set: VkDescriptorSet,
    reduce_ba_set: VkDescriptorSet,

    command_pool: VkCommandPool,
    command_buffer: VkCommandBuffer,
    fence: VkFence,

    buffers: SpinBuffers,
    readback: DynamicBuffer,
    pending: Option<PendingRun>,
    diagnostics: ComputeDiagnostics,
}

impl SpinCompute {
    pub fn new(device: &Device) -> Result<SpinCompute> {
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: std::ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: std::ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 2,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: std::ptr::null(),
            },
        ];
        let descriptor_info = VkDescriptorSetLayoutCreateInfo {
            bindingCount: bindings.len() as u32,
            pBindings: bindings.as_ptr(),
            ..Default::default()
        };

        let mut descriptor_layout: VkDescriptorSetLayout = VK_NULL_HANDLE;
        check("vkCreateDescriptorSetLayout", unsafe {
            (device.fns.create_descriptor_set_layout)(
                device.handle,
                &descriptor_info,
                NO_ALLOCATOR,
                &mut descriptor_layout,
            )
        })?;

        let push_range = VkPushConstantRange {
            stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: PUSH_BYTES,
        };
        let layout_info = VkPipelineLayoutCreateInfo {
            setLayoutCount: 1,
            pSetLayouts: &descriptor_layout,
            pushConstantRangeCount: 1,
            pPushConstantRanges: &push_range,
            ..Default::default()
        };

        let mut layout: VkPipelineLayout = VK_NULL_HANDLE;
        check("vkCreatePipelineLayout", unsafe {
            (device.fns.create_pipeline_layout)(
                device.handle,
                &layout_info,
                NO_ALLOCATOR,
                &mut layout,
            )
        })?;

        let reset_module = create_module(device, &compute_shader::reset())?;
        let evolve_module = create_module(device, &compute_shader::evolve())?;
        let signal_module = create_module(device, &compute_shader::signal_map())?;
        let reduce_module = create_module(device, &compute_shader::signal_reduce())?;

        let reset_pipeline = create_pipeline(device, layout, reset_module)?;
        let evolve_pipeline = create_pipeline(device, layout, evolve_module)?;
        let signal_pipeline = create_pipeline(device, layout, signal_module)?;
        let reduce_pipeline = create_pipeline(device, layout, reduce_module)?;

        unsafe {
            for module in [reset_module, evolve_module, signal_module, reduce_module] {
                (device.fns.destroy_shader_module)(device.handle, module, NO_ALLOCATOR);
            }
        }

        let pool_size = VkDescriptorPoolSize {
            type_: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount: 9,
        };
        let pool_info = VkDescriptorPoolCreateInfo {
            maxSets: 3,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
            ..Default::default()
        };

        let mut descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
        check("vkCreateDescriptorPool", unsafe {
            (device.fns.create_descriptor_pool)(
                device.handle,
                &pool_info,
                NO_ALLOCATOR,
                &mut descriptor_pool,
            )
        })?;

        let layouts = [descriptor_layout; 3];
        let set_allocation = VkDescriptorSetAllocateInfo {
            descriptorPool: descriptor_pool,
            descriptorSetCount: layouts.len() as u32,
            pSetLayouts: layouts.as_ptr(),
            ..Default::default()
        };
        let mut sets = [VK_NULL_HANDLE; 3];

        check("vkAllocateDescriptorSets", unsafe {
            (device.fns.allocate_descriptor_sets)(
                device.handle,
                &set_allocation,
                sets.as_mut_ptr(),
            )
        })?;

        let pool_info = VkCommandPoolCreateInfo {
            flags: VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            queueFamilyIndex: device.graphics_family,
            ..Default::default()
        };
        let mut command_pool: VkCommandPool = VK_NULL_HANDLE;
        check("vkCreateCommandPool", unsafe {
            (device.fns.create_command_pool)(
                device.handle,
                &pool_info,
                NO_ALLOCATOR,
                &mut command_pool,
            )
        })?;

        let command_allocation = VkCommandBufferAllocateInfo {
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
            ..Default::default()
        };
        let mut command_buffer: VkCommandBuffer = std::ptr::null_mut();
        check("vkAllocateCommandBuffers", unsafe {
            (device.fns.allocate_command_buffers)(
                device.handle,
                &command_allocation,
                &mut command_buffer,
            )
        })?;

        let fence_info = VkFenceCreateInfo {
            flags: VK_FENCE_CREATE_SIGNALED_BIT,
            ..Default::default()
        };
        let mut fence: VkFence = VK_NULL_HANDLE;
        check("vkCreateFence", unsafe {
            (device.fns.create_fence)(
                device.handle,
                &fence_info,
                NO_ALLOCATOR,
                &mut fence,
            )
        })?;

        let buffers = SpinBuffers::new(device, WORKGROUP_SIZE as usize)?;
        let readback =
            DynamicBuffer::new(device, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT)?;

        let mut compute = SpinCompute {
            descriptor_layout,
            layout,
            reset_pipeline,
            evolve_pipeline,
            signal_pipeline,
            reduce_pipeline,
            descriptor_pool,
            main_set: sets[0],
            reduce_ab_set: sets[1],
            reduce_ba_set: sets[2],
            command_pool,
            command_buffer,
            fence,
            buffers,
            readback,
            pending: None,
            diagnostics: ComputeDiagnostics {
                workgroup_size: WORKGROUP_SIZE,
                separate_queue: device.compute_queue_index != 0,
                ..Default::default()
            },
        };

        compute.write_descriptors(device);
        let verified = compute.verify(device)?;
        compute.diagnostics.verification_spins = verified.verification_spins;
        compute.diagnostics.verification_dispatches = verified.verification_dispatches;
        compute.diagnostics.verification_max_error = verified.verification_max_error;
        compute.diagnostics.signal_verification_max_error =
            compute.verify_signal(device)?;

        crate::log_info!(
            "compute",
            "Bloch evolve verified on {} spins, {} dispatches, max error {:.3e}",
            compute.diagnostics.verification_spins,
            compute.diagnostics.verification_dispatches,
            compute.diagnostics.verification_max_error
        );
        crate::log_info!(
            "compute",
            "full signal verified, max error {:.3e}",
            compute.diagnostics.signal_verification_max_error
        );
        crate::log_info!(
            "compute",
            "{} compute queue",
            if compute.diagnostics.separate_queue { "separate" } else { "shared" }
        );

        Ok(compute)
    }

    pub fn diagnostics(&self) -> ComputeDiagnostics {
        self.diagnostics
    }

    pub fn is_running(&self) -> bool {
        self.pending.is_some()
    }

    /// Records and submits one complete simulation without waiting for it.
    ///
    /// Returns false while another request is in flight. The caller retains its
    /// latest desired configuration and retries after poll returns the previous
    /// result.
    pub fn request_signal(
        &mut self,
        device: &Device,
        config: SimulationConfig,
        program: &SequenceProgram,
        sequence_revision: u64,
    ) -> Result<bool> {
        if self.pending.is_some() {
            return Ok(false);
        }

        let config = config.sanitized();
        let sequence = CompiledSequence::compile(&config, program);
        let spins = build_ensemble(&config);
        let padded = storage_capacity(spins.len());
        self.ensure_capacity(device, padded)?;

        let mut observations = Vec::with_capacity(config.observation_count + 4);
        observations.push((0.0f32, false));

        let mut time_s = 0.0f32;
        for step in sequence.steps() {
            time_s += step.duration_s;
            if step.observe_after {
                observations.push((time_s, step.acquired));
            }
        }

        let partials = padded.min(SIGNAL_PARTIALS);
        let visual_spins = spins.len().min(VISUAL_SPINS);
        let partial_bytes = partials * std::mem::size_of::<SignalPartial>();
        let snapshot_bytes = visual_spins * std::mem::size_of::<SpinState>();
        let signal_bytes = observations.len() * partial_bytes;
        let readback_bytes = signal_bytes + observations.len() * snapshot_bytes;

        self.readback.reserve(device, readback_bytes)?;
        self.upload_spins(device, &spins, padded)?;

        check("vkResetCommandPool", unsafe {
            (device.fns.reset_command_pool)(device.handle, self.command_pool, 0)
        })?;

        let begin = VkCommandBufferBeginInfo {
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ..Default::default()
        };
        check("vkBeginCommandBuffer", unsafe {
            (device.fns.begin_command_buffer)(self.command_buffer, &begin)
        })?;

        let groups = padded as u32 / WORKGROUP_SIZE;
        let mut dispatches = 1u32;

        unsafe {
            record_buffer_barrier(
                device,
                self.command_buffer,
                self.buffers.params.handle,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_HOST_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
            );

            (device.fns.cmd_bind_pipeline)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                self.reset_pipeline,
            );
            (device.fns.cmd_bind_descriptor_sets)(
                self.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                self.layout,
                0,
                1,
                &self.main_set,
                0,
                std::ptr::null(),
            );
            (device.fns.cmd_dispatch)(self.command_buffer, groups, 1, 1);

            record_state_for_observation(
                device,
                self.command_buffer,
                self.buffers.states.handle,
            );

            dispatches += record_observation(
                device,
                self.command_buffer,
                self.layout,
                self.signal_pipeline,
                self.reduce_pipeline,
                self.main_set,
                self.reduce_ab_set,
                self.reduce_ba_set,
                self.buffers.states.handle,
                self.buffers.scratch_a.handle,
                self.buffers.scratch_b.handle,
                self.readback.buffer.handle,
                padded,
                partials,
                visual_spins,
                signal_bytes,
                0,
            );

            let mut sample_index = 1usize;
            for step in sequence.steps() {
                let push = StepPush {
                    field: [
                        step.b1_t[0],
                        step.b1_t[1],
                        step.duration_s,
                        config.gamma_rad_s_t,
                    ],
                    gradient: [
                        step.gradient_t_m[0],
                        step.gradient_t_m[1],
                        step.gradient_t_m[2],
                        0.0,
                    ],
                };

                (device.fns.cmd_bind_pipeline)(
                    self.command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.evolve_pipeline,
                );
                (device.fns.cmd_bind_descriptor_sets)(
                    self.command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.layout,
                    0,
                    1,
                    &self.main_set,
                    0,
                    std::ptr::null(),
                );
                (device.fns.cmd_push_constants)(
                    self.command_buffer,
                    self.layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    PUSH_BYTES,
                    &push as *const _ as *const c_void,
                );
                (device.fns.cmd_dispatch)(self.command_buffer, groups, 1, 1);
                dispatches += 1;

                if step.observe_after {
                    record_state_for_observation(
                        device,
                        self.command_buffer,
                        self.buffers.states.handle,
                    );

                    dispatches += record_observation(
                        device,
                        self.command_buffer,
                        self.layout,
                        self.signal_pipeline,
                        self.reduce_pipeline,
                        self.main_set,
                        self.reduce_ab_set,
                        self.reduce_ba_set,
                        self.buffers.states.handle,
                        self.buffers.scratch_a.handle,
                        self.buffers.scratch_b.handle,
                        self.readback.buffer.handle,
                        padded,
                        partials,
                        visual_spins,
                        signal_bytes,
                        sample_index,
                    );
                    sample_index += 1;
                } else {
                    record_buffer_barrier(
                        device,
                        self.command_buffer,
                        self.buffers.states.handle,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    );
                }
            }

            record_buffer_barrier(
                device,
                self.command_buffer,
                self.readback.buffer.handle,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT,
            );
        }

        check("vkEndCommandBuffer", unsafe {
            (device.fns.end_command_buffer)(self.command_buffer)
        })?;
        check("vkResetFences", unsafe {
            (device.fns.reset_fences)(device.handle, 1, &self.fence)
        })?;

        let submit = VkSubmitInfo {
            commandBufferCount: 1,
            pCommandBuffers: &self.command_buffer,
            ..Default::default()
        };
        check("vkQueueSubmit", unsafe {
            (device.fns.queue_submit)(
                device.compute_queue,
                1,
                &submit,
                self.fence,
            )
        })?;

        self.diagnostics.model_spins = spins.len() as u32;
        self.diagnostics.model_steps = sequence.steps().len() as u32;
        self.diagnostics.signal_samples = observations.len() as u32;
        self.diagnostics.signal_dispatches = dispatches;
        self.diagnostics.partials_per_sample = partials as u32;
        self.diagnostics.visual_spins = visual_spins as u32;
        self.diagnostics.running = true;
        self.diagnostics.run_ms = 0.0;

        self.pending = Some(PendingRun {
            config,
            sequence_revision,
            observations,
            partials,
            visual_spins,
            signal_bytes,
            dispatches,
            started: Instant::now(),
        });

        Ok(true)
    }

    /// Polls the active request and publishes it after the compute fence signals.
    pub fn poll_signal(&mut self, device: &Device) -> Result<Option<SimulationTrace>> {
        if self.pending.is_none() {
            return Ok(None);
        }

        let status = unsafe { (device.fns.get_fence_status)(device.handle, self.fence) };
        if status == VK_NOT_READY {
            return Ok(None);
        }
        if status != VK_SUCCESS {
            return Err(vk_err("vkGetFenceStatus", status));
        }

        self.readback.buffer.invalidate(device)?;
        let pending = self.pending.take().expect("the pending compute run vanished");
        let partial_count = pending.observations.len() * pending.partials;
        let state_count = pending.observations.len() * pending.visual_spins;
        let mapped = self.readback.buffer.mapped;

        if mapped.is_null() {
            return Err(Error::vulkan("compute readback buffer is not mapped"));
        }

        let partials = unsafe {
            std::slice::from_raw_parts(
                mapped as *const SignalPartial,
                partial_count,
            )
        };
        let states = unsafe {
            std::slice::from_raw_parts(
                mapped.add(pending.signal_bytes) as *const SpinState,
                state_count,
            )
        };

        let mut samples = Vec::with_capacity(pending.observations.len());
        for (sample_index, &(time_s, acquired)) in
            pending.observations.iter().enumerate()
        {
            let start = sample_index * pending.partials;
            let end = start + pending.partials;
            let mut real = 0.0f32;
            let mut imaginary = 0.0f32;

            for partial in &partials[start..end] {
                real += partial.value[0];
                imaginary += partial.value[1];
            }

            samples.push(SignalSample {
                time_s,
                real,
                imaginary,
                magnitude: (real * real + imaginary * imaginary).sqrt(),
                acquired,
            });
        }

        self.diagnostics.running = false;
        self.diagnostics.signal_dispatches = pending.dispatches;
        self.diagnostics.run_ms = pending.started.elapsed().as_secs_f32() * 1_000.0;

        Ok(Some(SimulationTrace {
            config: pending.config,
            sequence_revision: pending.sequence_revision,
            samples,
            visual_spin_count: pending.visual_spins,
            visual_states: states.to_vec(),
        }))
    }

    /// Runs reset and field evolution synchronously for startup verification.
    pub fn run_final(
        &mut self,
        device: &Device,
        spins: &[SpinParams],
        steps: &[FieldStep],
        gamma_rad_s_t: f32,
    ) -> Result<Vec<SpinState>> {
        if spins.is_empty() {
            return Ok(Vec::new());
        }

        let padded = storage_capacity(spins.len());
        self.ensure_capacity(device, padded)?;
        self.upload_spins(device, spins, padded)?;

        let groups = padded as u32 / WORKGROUP_SIZE;
        let params_buffer = self.buffers.params.handle;
        let state_buffer = self.buffers.states.handle;
        let main_set = self.main_set;
        let layout = self.layout;
        let reset_pipeline = self.reset_pipeline;
        let evolve_pipeline = self.evolve_pipeline;

        device.one_time_submit(|command_buffer| unsafe {
            record_buffer_barrier(
                device,
                command_buffer,
                params_buffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_HOST_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
            );

            (device.fns.cmd_bind_pipeline)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                reset_pipeline,
            );
            (device.fns.cmd_bind_descriptor_sets)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                layout,
                0,
                1,
                &main_set,
                0,
                std::ptr::null(),
            );
            (device.fns.cmd_dispatch)(command_buffer, groups, 1, 1);

            record_buffer_barrier(
                device,
                command_buffer,
                state_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            );

            if !steps.is_empty() {
                (device.fns.cmd_bind_pipeline)(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    evolve_pipeline,
                );

                for step in steps {
                    let push = StepPush {
                        field: [
                            step.b1_t[0],
                            step.b1_t[1],
                            step.duration_s,
                            gamma_rad_s_t,
                        ],
                        gradient: [
                            step.gradient_t_m[0],
                            step.gradient_t_m[1],
                            step.gradient_t_m[2],
                            0.0,
                        ],
                    };
                    (device.fns.cmd_push_constants)(
                        command_buffer,
                        layout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        PUSH_BYTES,
                        &push as *const _ as *const c_void,
                    );
                    (device.fns.cmd_dispatch)(command_buffer, groups, 1, 1);

                    record_buffer_barrier(
                        device,
                        command_buffer,
                        state_buffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    );
                }
            }

            record_buffer_barrier(
                device,
                command_buffer,
                state_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT,
            );
        })?;

        self.buffers.states.invalidate(device)?;
        let mapped = self.buffers.states.mapped as *const SpinState;
        if mapped.is_null() {
            return Err(Error::vulkan("compute state buffer is not mapped"));
        }

        Ok(unsafe { std::slice::from_raw_parts(mapped, spins.len()) }.to_vec())
    }

    fn verify(&mut self, device: &Device) -> Result<ComputeDiagnostics> {
        const SPINS: usize = 17;
        const PULSE_STEPS: usize = 48;

        let mut params = Vec::with_capacity(SPINS);
        for index in 0..SPINS {
            let coordinate = index as f32 / (SPINS - 1) as f32 - 0.5;
            params.push(SpinParams::new(
                [coordinate * 0.08, coordinate * -0.03, 0.0],
                1.0,
                0.75 + index as f32 * 0.013,
                0.09 + index as f32 * 0.001,
                -90.0 + index as f32 * 11.25,
                1.0 / SPINS as f32,
            ));
        }

        let pulse_dt = 10.0e-6;
        let pulse_duration = pulse_dt * PULSE_STEPS as f32;
        let b1 = PI * 0.5 / (PROTON_GAMMA_RAD * pulse_duration);
        let mut steps = Vec::with_capacity(PULSE_STEPS + 2);

        for _ in 0..PULSE_STEPS {
            steps.push(FieldStep {
                duration_s: pulse_dt,
                b1_t: [b1, 0.0],
                gradient_t_m: [0.0; 3],
                observe_after: false,
                acquired: false,
            });
        }
        steps.push(FieldStep {
            duration_s: 1.7e-3,
            b1_t: [0.0; 2],
            gradient_t_m: [0.16e-3, -0.04e-3, 0.02e-3],
            observe_after: false,
            acquired: false,
        });
        steps.push(FieldStep {
            duration_s: 2.1e-3,
            b1_t: [0.0; 2],
            gradient_t_m: [-0.07e-3, 0.02e-3, 0.0],
            observe_after: false,
            acquired: false,
        });

        let mut expected: Vec<SpinState> =
            params.iter().map(SpinState::equilibrium).collect();
        for step in &steps {
            for (spin, state) in params.iter().zip(expected.iter_mut()) {
                advance_spin(spin, state, step, PROTON_GAMMA_RAD);
            }
        }

        let actual = self.run_final(device, &params, &steps, PROTON_GAMMA_RAD)?;
        let mut max_error = 0.0f32;

        for (cpu, gpu) in expected.iter().zip(&actual) {
            for component in 0..3 {
                max_error = max_error.max(
                    (cpu.magnetization[component]
                        - gpu.magnetization[component])
                        .abs(),
                );
            }
        }

        if !max_error.is_finite() || max_error > 2.5e-3 {
            return Err(Error::vulkan(format!(
                "compute Bloch verification failed, max error {:.6}",
                max_error
            )));
        }

        Ok(ComputeDiagnostics {
            verification_spins: SPINS as u32,
            verification_dispatches: steps.len() as u32 + 1,
            verification_max_error: max_error,
            ..Default::default()
        })
    }

    /// Compares the complete reduction path against the CPU reference.
    ///
    /// The sequence contains a shaped RF pulse, overlapping gradients and two
    /// ADC blocks. This covers timeline compilation, evolution, signal mapping,
    /// reduction, transfer readback and observation ordering.
    fn verify_signal(&mut self, device: &Device) -> Result<f32> {
        let config = SimulationConfig {
            sequence: SequenceKind::Custom,
            spin_count: 257,
            observation_count: 113,
            t1_s: 0.83,
            t2_s: 0.11,
            t1_spread: 0.17,
            t2_spread: 0.23,
            center_offset_hz: -13.0,
            offset_span_hz: 180.0,
            te_s: 0.042,
            ..SimulationConfig::default()
        };

        let mut program = SequenceProgram::new(0.072, 0.042);
        program.events.push(SequenceEvent::rf(
            0.003,
            0.9e-3,
            PI * 0.5,
            0.31,
            RfShape::Gaussian,
            32,
        ));
        program.events.push(SequenceEvent::gradient(
            0.010,
            0.008,
            [0.21e-3, -0.04e-3, 0.02e-3],
        ));
        program.events.push(SequenceEvent::gradient(
            0.014,
            0.011,
            [-0.07e-3, 0.09e-3, 0.0],
        ));
        program.events.push(SequenceEvent::adc(0.030, 0.008));
        program.events.push(SequenceEvent::adc(0.046, 0.010));

        let expected = run_program(config, &program);
        if !self.request_signal(device, config, &program, 0xA11CE)? {
            return Err(Error::vulkan(
                "signal verification could not submit its model",
            ));
        }

        check("vkWaitForFences", unsafe {
            (device.fns.wait_for_fences)(
                device.handle,
                1,
                &self.fence,
                VK_TRUE,
                u64::MAX,
            )
        })?;

        let actual = self
            .poll_signal(device)?
            .ok_or_else(|| Error::vulkan("signal verification produced no result"))?;

        if expected.samples().len() != actual.samples.len() {
            return Err(Error::vulkan(format!(
                "signal verification sample count differs: CPU {} GPU {}",
                expected.samples().len(),
                actual.samples.len()
            )));
        }

        let mut max_error = 0.0f32;
        for (cpu, gpu) in expected.samples().iter().zip(&actual.samples) {
            max_error = max_error
                .max((cpu.real - gpu.real).abs())
                .max((cpu.imaginary - gpu.imaginary).abs())
                .max((cpu.magnitude - gpu.magnitude).abs());
        }

        if !max_error.is_finite() || max_error > 3.0e-3 {
            return Err(Error::vulkan(format!(
                "full signal verification failed, max error {:.6}",
                max_error
            )));
        }

        self.diagnostics.model_spins = 0;
        self.diagnostics.model_steps = 0;
        self.diagnostics.signal_samples = 0;
        self.diagnostics.signal_dispatches = 0;
        self.diagnostics.partials_per_sample = 0;
        self.diagnostics.visual_spins = 0;
        self.diagnostics.run_ms = 0.0;
        self.diagnostics.running = false;
        Ok(max_error)
    }

    fn upload_spins(
        &self,
        device: &Device,
        spins: &[SpinParams],
        padded: usize,
    ) -> Result<()> {
        let dummy = SpinParams::new([0.0; 3], 0.0, 1.0, 1.0, 0.0, 0.0);
        let mut upload = vec![dummy; padded];
        upload[..spins.len()].copy_from_slice(spins);

        self.buffers.params.write(0, as_bytes(&upload));
        self.buffers.params.flush(device)
    }

    fn ensure_capacity(&mut self, device: &Device, capacity: usize) -> Result<()> {
        if capacity <= self.buffers.capacity {
            return Ok(());
        }

        if self.pending.is_some() {
            return Err(Error::vulkan(
                "cannot resize compute storage while a model is running",
            ));
        }

        device.wait_idle()?;
        let next = SpinBuffers::new(device, capacity)?;
        let mut old = std::mem::replace(&mut self.buffers, next);
        old.destroy(device);
        self.write_descriptors(device);

        let bytes_per_spin = std::mem::size_of::<SpinParams>()
            + std::mem::size_of::<SpinState>()
            + std::mem::size_of::<SignalPartial>() * 2;

        crate::log_info!(
            "compute",
            "spin storage grew to {} records, {} MiB",
            capacity,
            capacity * bytes_per_spin / (1024 * 1024)
        );
        Ok(())
    }

    fn write_descriptors(&self, device: &Device) {
        let infos = [
            VkDescriptorBufferInfo {
                buffer: self.buffers.params.handle,
                offset: 0,
                range: self.buffers.params.size,
            },
            VkDescriptorBufferInfo {
                buffer: self.buffers.states.handle,
                offset: 0,
                range: self.buffers.states.size,
            },
            VkDescriptorBufferInfo {
                buffer: self.buffers.scratch_a.handle,
                offset: 0,
                range: self.buffers.scratch_a.size,
            },
            VkDescriptorBufferInfo {
                buffer: self.buffers.scratch_b.handle,
                offset: 0,
                range: self.buffers.scratch_b.size,
            },
        ];

        let entries = [
            (self.main_set, 0, 0usize),
            (self.main_set, 1, 1usize),
            (self.main_set, 2, 2usize),
            (self.reduce_ab_set, 0, 2usize),
            (self.reduce_ab_set, 1, 3usize),
            (self.reduce_ab_set, 2, 2usize),
            (self.reduce_ba_set, 0, 3usize),
            (self.reduce_ba_set, 1, 2usize),
            (self.reduce_ba_set, 2, 3usize),
        ];

        let mut writes = Vec::with_capacity(entries.len());
        for &(set, binding, info) in &entries {
            writes.push(VkWriteDescriptorSet {
                dstSet: set,
                dstBinding: binding,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo: &infos[info],
                ..Default::default()
            });
        }

        unsafe {
            (device.fns.update_descriptor_sets)(
                device.handle,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        if self.pending.is_some() {
            let _ = unsafe {
                (device.fns.wait_for_fences)(
                    device.handle,
                    1,
                    &self.fence,
                    VK_TRUE,
                    u64::MAX,
                )
            };
            self.pending = None;
        }

        self.buffers.destroy(device);
        self.readback.destroy(device);

        unsafe {
            (device.fns.destroy_fence)(device.handle, self.fence, NO_ALLOCATOR);
            (device.fns.destroy_command_pool)(
                device.handle,
                self.command_pool,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_descriptor_pool)(
                device.handle,
                self.descriptor_pool,
                NO_ALLOCATOR,
            );

            for pipeline in [
                self.reset_pipeline,
                self.evolve_pipeline,
                self.signal_pipeline,
                self.reduce_pipeline,
            ] {
                (device.fns.destroy_pipeline)(device.handle, pipeline, NO_ALLOCATOR);
            }

            (device.fns.destroy_pipeline_layout)(
                device.handle,
                self.layout,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_descriptor_set_layout)(
                device.handle,
                self.descriptor_layout,
                NO_ALLOCATOR,
            );
        }

        self.fence = VK_NULL_HANDLE;
        self.command_pool = VK_NULL_HANDLE;
        self.descriptor_pool = VK_NULL_HANDLE;
        self.reset_pipeline = VK_NULL_HANDLE;
        self.evolve_pipeline = VK_NULL_HANDLE;
        self.signal_pipeline = VK_NULL_HANDLE;
        self.reduce_pipeline = VK_NULL_HANDLE;
        self.layout = VK_NULL_HANDLE;
        self.descriptor_layout = VK_NULL_HANDLE;
    }
}


unsafe fn record_state_for_observation(
    device: &Device,
    command_buffer: VkCommandBuffer,
    state_buffer: VkBuffer,
) {
    record_buffer_barrier(
        device,
        command_buffer,
        state_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_observation(
    device: &Device,
    command_buffer: VkCommandBuffer,
    layout: VkPipelineLayout,
    signal_pipeline: VkPipeline,
    reduce_pipeline: VkPipeline,
    main_set: VkDescriptorSet,
    reduce_ab_set: VkDescriptorSet,
    reduce_ba_set: VkDescriptorSet,
    state_buffer: VkBuffer,
    scratch_a: VkBuffer,
    scratch_b: VkBuffer,
    readback: VkBuffer,
    padded: usize,
    partials: usize,
    visual_spins: usize,
    signal_bytes: usize,
    sample_index: usize,
) -> u32 {
    let dispatches = record_signal_sample(
        device,
        command_buffer,
        layout,
        signal_pipeline,
        reduce_pipeline,
        main_set,
        reduce_ab_set,
        reduce_ba_set,
        scratch_a,
        scratch_b,
        readback,
        padded,
        partials,
        sample_index,
    );

    let snapshot_bytes =
        (visual_spins * std::mem::size_of::<SpinState>()) as VkDeviceSize;
    let snapshot_offset = signal_bytes as VkDeviceSize
        + sample_index as VkDeviceSize * snapshot_bytes;
    let copy = VkBufferCopy {
        srcOffset: 0,
        dstOffset: snapshot_offset,
        size: snapshot_bytes,
    };

    (device.fns.cmd_copy_buffer)(
        command_buffer,
        state_buffer,
        readback,
        1,
        &copy,
    );

    record_buffer_barrier(
        device,
        command_buffer,
        state_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT
            | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    );

    dispatches
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_signal_sample(
    device: &Device,
    command_buffer: VkCommandBuffer,
    layout: VkPipelineLayout,
    signal_pipeline: VkPipeline,
    reduce_pipeline: VkPipeline,
    main_set: VkDescriptorSet,
    reduce_ab_set: VkDescriptorSet,
    reduce_ba_set: VkDescriptorSet,
    scratch_a: VkBuffer,
    scratch_b: VkBuffer,
    readback: VkBuffer,
    padded: usize,
    partials: usize,
    sample_index: usize,
) -> u32 {
    let scratch_barriers = [
        buffer_memory_barrier(
            scratch_a,
            VK_ACCESS_SHADER_READ_BIT
                | VK_ACCESS_SHADER_WRITE_BIT
                | VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
        ),
        buffer_memory_barrier(
            scratch_b,
            VK_ACCESS_SHADER_READ_BIT
                | VK_ACCESS_SHADER_WRITE_BIT
                | VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
        ),
    ];
    (device.fns.cmd_pipeline_barrier)(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        std::ptr::null(),
        scratch_barriers.len() as u32,
        scratch_barriers.as_ptr() as *const c_void,
        0,
        std::ptr::null(),
    );

    (device.fns.cmd_bind_pipeline)(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        signal_pipeline,
    );
    (device.fns.cmd_bind_descriptor_sets)(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        layout,
        0,
        1,
        &main_set,
        0,
        std::ptr::null(),
    );
    (device.fns.cmd_dispatch)(
        command_buffer,
        padded as u32 / WORKGROUP_SIZE,
        1,
        1,
    );

    record_scratch_dependency(device, command_buffer, scratch_a, scratch_b);

    let mut count = padded;
    let mut source_is_a = true;
    let mut dispatches = 1u32;

    while count > partials {
        let output_count = count / 2;
        let set = if source_is_a {
            reduce_ab_set
        } else {
            reduce_ba_set
        };

        (device.fns.cmd_bind_pipeline)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            reduce_pipeline,
        );
        (device.fns.cmd_bind_descriptor_sets)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            layout,
            0,
            1,
            &set,
            0,
            std::ptr::null(),
        );
        (device.fns.cmd_dispatch)(
            command_buffer,
            output_count as u32 / WORKGROUP_SIZE,
            1,
            1,
        );

        dispatches += 1;
        count = output_count;
        source_is_a = !source_is_a;
        record_scratch_dependency(device, command_buffer, scratch_a, scratch_b);
    }

    let source = if source_is_a { scratch_a } else { scratch_b };
    record_buffer_barrier(
        device,
        command_buffer,
        source,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
    );

    let bytes =
        (partials * std::mem::size_of::<SignalPartial>()) as VkDeviceSize;
    let copy = VkBufferCopy {
        srcOffset: 0,
        dstOffset: sample_index as VkDeviceSize * bytes,
        size: bytes,
    };
    (device.fns.cmd_copy_buffer)(command_buffer, source, readback, 1, &copy);

    dispatches
}

unsafe fn record_scratch_dependency(
    device: &Device,
    command_buffer: VkCommandBuffer,
    scratch_a: VkBuffer,
    scratch_b: VkBuffer,
) {
    let barriers = [
        buffer_memory_barrier(
            scratch_a,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT
                | VK_ACCESS_SHADER_WRITE_BIT
                | VK_ACCESS_TRANSFER_READ_BIT,
        ),
        buffer_memory_barrier(
            scratch_b,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT
                | VK_ACCESS_SHADER_WRITE_BIT
                | VK_ACCESS_TRANSFER_READ_BIT,
        ),
    ];

    (device.fns.cmd_pipeline_barrier)(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        std::ptr::null(),
        barriers.len() as u32,
        barriers.as_ptr() as *const c_void,
        0,
        std::ptr::null(),
    );
}

unsafe fn record_buffer_barrier(
    device: &Device,
    command_buffer: VkCommandBuffer,
    buffer: VkBuffer,
    source_stage: VkPipelineStageFlags,
    target_stage: VkPipelineStageFlags,
    source_access: VkAccessFlags,
    target_access: VkAccessFlags,
) {
    let barrier = buffer_memory_barrier(buffer, source_access, target_access);
    (device.fns.cmd_pipeline_barrier)(
        command_buffer,
        source_stage,
        target_stage,
        0,
        0,
        std::ptr::null(),
        1,
        &barrier as *const _ as *const c_void,
        0,
        std::ptr::null(),
    );
}

fn buffer_memory_barrier(
    buffer: VkBuffer,
    source_access: VkAccessFlags,
    target_access: VkAccessFlags,
) -> VkBufferMemoryBarrier {
    VkBufferMemoryBarrier {
        srcAccessMask: source_access,
        dstAccessMask: target_access,
        srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
        dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
        buffer,
        offset: 0,
        size: VK_WHOLE_SIZE,
        ..Default::default()
    }
}

fn storage_capacity(count: usize) -> usize {
    count.max(WORKGROUP_SIZE as usize).next_power_of_two()
}

fn create_module(device: &Device, words: &[u32]) -> Result<VkShaderModule> {
    let info = VkShaderModuleCreateInfo {
        codeSize: std::mem::size_of_val(words),
        pCode: words.as_ptr(),
        ..Default::default()
    };
    let mut module: VkShaderModule = VK_NULL_HANDLE;
    check("vkCreateShaderModule", unsafe {
        (device.fns.create_shader_module)(
            device.handle,
            &info,
            NO_ALLOCATOR,
            &mut module,
        )
    })?;
    Ok(module)
}

fn create_pipeline(
    device: &Device,
    layout: VkPipelineLayout,
    module: VkShaderModule,
) -> Result<VkPipeline> {
    let stage = VkPipelineShaderStageCreateInfo {
        stage: VK_SHADER_STAGE_COMPUTE_BIT,
        module,
        pName: b"main\0".as_ptr() as *const c_char,
        ..Default::default()
    };
    let info = VkComputePipelineCreateInfo {
        stage,
        layout,
        basePipelineIndex: -1,
        ..Default::default()
    };
    let mut pipeline: VkPipeline = VK_NULL_HANDLE;
    check("vkCreateComputePipelines", unsafe {
        (device.fns.create_compute_pipelines)(
            device.handle,
            VK_NULL_HANDLE,
            1,
            &info,
            NO_ALLOCATOR,
            &mut pipeline,
        )
    })?;
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_layout_matches_shader_block() {
        assert_eq!(std::mem::size_of::<StepPush>(), PUSH_BYTES as usize);
        assert_eq!(std::mem::size_of::<SignalPartial>(), 16);
    }

    #[test]
    fn storage_capacity_is_a_power_of_two() {
        assert_eq!(storage_capacity(1), 64);
        assert_eq!(storage_capacity(64), 64);
        assert_eq!(storage_capacity(65), 128);
        assert_eq!(storage_capacity(65_536), 65_536);
    }

    #[test]
    fn snapshot_storage_is_frame_major() {
        let trace = SimulationTrace {
            config: SimulationConfig::default(),
            sequence_revision: 1,
            samples: vec![
                SignalSample { time_s: 0.0, ..Default::default() },
                SignalSample { time_s: 1.0, ..Default::default() },
            ],
            visual_spin_count: 2,
            visual_states: vec![
                SpinState::new(1.0, 0.0, 0.0),
                SpinState::new(2.0, 0.0, 0.0),
                SpinState::new(3.0, 0.0, 0.0),
                SpinState::new(4.0, 0.0, 0.0),
            ],
        };

        assert_eq!(trace.states_at(0.0)[0].mx(), 1.0);
        assert_eq!(trace.states_at(1.0)[0].mx(), 3.0);
    }
}