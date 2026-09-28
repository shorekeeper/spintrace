//! Direct Vulkan rendering of magnetization vectors and phase bins.
//!
//! One host visible state buffer and one device local histogram buffer belong to
//! every frame slot. The frame fence protects both. Selected visualization
//! states are copied into the slot after the fence signals, then a compute pass
//! clears and rebuilds the phase histogram before the render pass begins.
//!
//! Vector and histogram geometry is procedural. A six-vertex corner buffer is
//! the only vertex buffer: the vector pipeline uses its first two entries as
//! endpoint factors, while the histogram pipeline uses all six as a unit quad.
//! Magnetization and bin counts are read through storage descriptors.
//!
//! The state upload is a visualization cache, not a simulation path. The source
//! records come from the completed Vulkan simulation trace. No vector or bar
//! geometry is generated on the CPU or appended to the GUI DrawList.

use std::ffi::{c_char, c_void};

use crate::core::Result;
use crate::render::batch::Rect;
use crate::render::device::Device;
use crate::render::memory::{as_bytes, Buffer, Location};
use crate::render::spin_view_shader::{
    self, HISTOGRAM_GROUPS, PHASE_BINS, VECTOR_INSTANCES, VISUAL_SPINS,
};
use crate::render::vk::*;
use crate::sim::SpinState;

const PUSH_BYTES: u32 = 32;

#[derive(Clone, Copy)]
pub struct SpinVisualFrame<'a> {
    pub states: &'a [SpinState],
    pub vectors: Rect,
    pub histogram: Option<Rect>,
    pub vector_scale: f32,
}

struct VisualSlot {
    states: Buffer,
    histogram: Buffer,
    descriptor: VkDescriptorSet,
}

pub struct SpinView {
    descriptor_layout: VkDescriptorSetLayout,
    layout: VkPipelineLayout,
    histogram_pipeline: VkPipeline,
    vector_pipeline: VkPipeline,
    bar_pipeline: VkPipeline,
    descriptor_pool: VkDescriptorPool,
    corners: Buffer,
    slots: Vec<VisualSlot>,
    upload: Vec<SpinState>,
}

impl SpinView {
    pub fn new(
        device: &Device,
        render_pass: VkRenderPass,
        frame_count: usize,
    ) -> Result<SpinView> {
        let bindings = [
            VkDescriptorSetLayoutBinding {
                binding: 0,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT,
                pImmutableSamplers: std::ptr::null(),
            },
            VkDescriptorSetLayoutBinding {
                binding: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT,
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
            stageFlags: VK_SHADER_STAGE_VERTEX_BIT,
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

        let histogram_module =
            create_module(device, &spin_view_shader::phase_histogram())?;
        let vector_module =
            create_module(device, &spin_view_shader::vectors_vertex())?;
        let bar_module =
            create_module(device, &spin_view_shader::histogram_vertex())?;
        let fragment_module =
            create_module(device, &spin_view_shader::solid_fragment())?;

        let histogram_pipeline =
            create_compute_pipeline(device, layout, histogram_module)?;
        let vector_pipeline = create_graphics_pipeline(
            device,
            render_pass,
            layout,
            vector_module,
            fragment_module,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        )?;
        let bar_pipeline = create_graphics_pipeline(
            device,
            render_pass,
            layout,
            bar_module,
            fragment_module,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        )?;

        unsafe {
            for module in [
                histogram_module,
                vector_module,
                bar_module,
                fragment_module,
            ] {
                (device.fns.destroy_shader_module)(
                    device.handle,
                    module,
                    NO_ALLOCATOR,
                );
            }
        }

        let frame_count = frame_count.max(1);
        let pool_size = VkDescriptorPoolSize {
            type_: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount: frame_count as u32 * 2,
        };
        let pool_info = VkDescriptorPoolCreateInfo {
            maxSets: frame_count as u32,
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

        let layouts = vec![descriptor_layout; frame_count];
        let allocation = VkDescriptorSetAllocateInfo {
            descriptorPool: descriptor_pool,
            descriptorSetCount: layouts.len() as u32,
            pSetLayouts: layouts.as_ptr(),
            ..Default::default()
        };
        let mut sets = vec![VK_NULL_HANDLE; frame_count];
        check("vkAllocateDescriptorSets", unsafe {
            (device.fns.allocate_descriptor_sets)(
                device.handle,
                &allocation,
                sets.as_mut_ptr(),
            )
        })?;

        let mut slots = Vec::with_capacity(frame_count);
        for descriptor in sets {
            let states = Buffer::new(
                device,
                (VISUAL_SPINS * std::mem::size_of::<SpinState>()) as VkDeviceSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                Location::HostVisible,
            )?;
            let histogram = Buffer::new(
                device,
                (PHASE_BINS * std::mem::size_of::<u32>()) as VkDeviceSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                Location::DeviceLocal,
            )?;

            let infos = [
                VkDescriptorBufferInfo {
                    buffer: states.handle,
                    offset: 0,
                    range: states.size,
                },
                VkDescriptorBufferInfo {
                    buffer: histogram.handle,
                    offset: 0,
                    range: histogram.size,
                },
            ];
            let writes = [
                VkWriteDescriptorSet {
                    dstSet: descriptor,
                    dstBinding: 0,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo: &infos[0],
                    ..Default::default()
                },
                VkWriteDescriptorSet {
                    dstSet: descriptor,
                    dstBinding: 1,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo: &infos[1],
                    ..Default::default()
                },
            ];

            unsafe {
                (device.fns.update_descriptor_sets)(
                    device.handle,
                    writes.len() as u32,
                    writes.as_ptr(),
                    0,
                    std::ptr::null(),
                );
            }

            slots.push(VisualSlot { states, histogram, descriptor });
        }

        let corners_data: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let corners = Buffer::new(
            device,
            std::mem::size_of_val(&corners_data) as VkDeviceSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            Location::HostVisible,
        )?;
        corners.write(0, as_bytes(&corners_data));
        corners.flush(device)?;

        Ok(SpinView {
            descriptor_layout,
            layout,
            histogram_pipeline,
            vector_pipeline,
            bar_pipeline,
            descriptor_pool,
            corners,
            slots,
            upload: vec![SpinState::new(0.0, 0.0, 0.0); VISUAL_SPINS],
        })
    }

    /// Copies one selected snapshot into a protected frame slot.
    pub fn upload(
        &mut self,
        device: &Device,
        slot: usize,
        states: &[SpinState],
    ) -> Result<()> {
        for state in &mut self.upload {
            *state = SpinState::new(0.0, 0.0, 0.0);
        }

        for (target, source) in self.upload.iter_mut().zip(states.iter()) {
            *target = *source;
            target.magnetization[3] = 1.0;
        }

        let visual = &self.slots[slot];
        visual.states.write(0, as_bytes(&self.upload));
        visual.states.flush(device)
    }

    /// Clears and rebuilds the phase histogram before the render pass.
    pub unsafe fn record_prepass(
        &self,
        device: &Device,
        command_buffer: VkCommandBuffer,
        slot: usize,
    ) {
        let visual = &self.slots[slot];

        (device.fns.cmd_fill_buffer)(
            command_buffer,
            visual.histogram.handle,
            0,
            visual.histogram.size,
            0,
        );

        let state_ready = VkBufferMemoryBarrier {
            srcAccessMask: VK_ACCESS_HOST_WRITE_BIT,
            dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            buffer: visual.states.handle,
            offset: 0,
            size: VK_WHOLE_SIZE,
            ..Default::default()
        };
        let bins_ready = VkBufferMemoryBarrier {
            srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            buffer: visual.histogram.handle,
            offset: 0,
            size: VK_WHOLE_SIZE,
            ..Default::default()
        };

        (device.fns.cmd_pipeline_barrier)(
            command_buffer,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            std::ptr::null(),
            2,
            [state_ready, bins_ready].as_ptr() as *const c_void,
            0,
            std::ptr::null(),
        );

        (device.fns.cmd_bind_pipeline)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.histogram_pipeline,
        );
        (device.fns.cmd_bind_descriptor_sets)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout,
            0,
            1,
            &visual.descriptor,
            0,
            std::ptr::null(),
        );
        (device.fns.cmd_dispatch)(
            command_buffer,
            HISTOGRAM_GROUPS,
            1,
            1,
        );

        let bins_visible = VkBufferMemoryBarrier {
            srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            buffer: visual.histogram.handle,
            offset: 0,
            size: VK_WHOLE_SIZE,
            ..Default::default()
        };
        (device.fns.cmd_pipeline_barrier)(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0,
            0,
            std::ptr::null(),
            1,
            &bins_visible as *const _ as *const c_void,
            0,
            std::ptr::null(),
        );
    }

    /// Draws vectors and the optional histogram inside the active render pass.
    pub unsafe fn record_draw(
        &self,
        device: &Device,
        command_buffer: VkCommandBuffer,
        slot: usize,
        frame: SpinVisualFrame<'_>,
        extent: VkExtent2D,
    ) {
        let viewport = VkViewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            minDepth: 0.0,
            maxDepth: 1.0,
        };
        (device.fns.cmd_set_viewport)(command_buffer, 0, 1, &viewport);

        let buffers = [self.corners.handle];
        let offsets = [0u64];
        (device.fns.cmd_bind_vertex_buffers)(
            command_buffer,
            0,
            1,
            buffers.as_ptr(),
            offsets.as_ptr(),
        );
        (device.fns.cmd_bind_descriptor_sets)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.layout,
            0,
            1,
            &self.slots[slot].descriptor,
            0,
            std::ptr::null(),
        );

        if let Some(scissor) = scissor(frame.vectors, extent) {
            (device.fns.cmd_set_scissor)(command_buffer, 0, 1, &scissor);
            (device.fns.cmd_bind_pipeline)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.vector_pipeline,
            );

            let push = [
                frame.vectors.x,
                frame.vectors.y,
                frame.vectors.w,
                frame.vectors.h,
                extent.width as f32,
                extent.height as f32,
                frame.vector_scale,
                0.0,
            ];
            (device.fns.cmd_push_constants)(
                command_buffer,
                self.layout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                PUSH_BYTES,
                push.as_ptr() as *const c_void,
            );
            (device.fns.cmd_draw)(
                command_buffer,
                2,
                VECTOR_INSTANCES,
                0,
                0,
            );
        }

        if let Some(histogram) = frame.histogram {
            if let Some(scissor) = scissor(histogram, extent) {
                (device.fns.cmd_set_scissor)(command_buffer, 0, 1, &scissor);
                (device.fns.cmd_bind_pipeline)(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.bar_pipeline,
                );

                let push = [
                    histogram.x,
                    histogram.y,
                    histogram.w,
                    histogram.h,
                    extent.width as f32,
                    extent.height as f32,
                    1.0,
                    0.0,
                ];
                (device.fns.cmd_push_constants)(
                    command_buffer,
                    self.layout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    PUSH_BYTES,
                    push.as_ptr() as *const c_void,
                );
                (device.fns.cmd_draw)(
                    command_buffer,
                    6,
                    PHASE_BINS as u32,
                    0,
                    0,
                );
            }
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        for slot in &mut self.slots {
            slot.states.destroy(device);
            slot.histogram.destroy(device);
        }
        self.slots.clear();
        self.corners.destroy(device);

        unsafe {
            (device.fns.destroy_descriptor_pool)(
                device.handle,
                self.descriptor_pool,
                NO_ALLOCATOR,
            );
            for pipeline in [
                self.histogram_pipeline,
                self.vector_pipeline,
                self.bar_pipeline,
            ] {
                (device.fns.destroy_pipeline)(
                    device.handle,
                    pipeline,
                    NO_ALLOCATOR,
                );
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

        self.descriptor_pool = VK_NULL_HANDLE;
        self.histogram_pipeline = VK_NULL_HANDLE;
        self.vector_pipeline = VK_NULL_HANDLE;
        self.bar_pipeline = VK_NULL_HANDLE;
        self.layout = VK_NULL_HANDLE;
        self.descriptor_layout = VK_NULL_HANDLE;
    }
}

fn scissor(rect: Rect, extent: VkExtent2D) -> Option<VkRect2D> {
    let x = rect.x.floor().max(0.0).min(extent.width as f32) as i32;
    let y = rect.y.floor().max(0.0).min(extent.height as f32) as i32;
    let right = rect
        .right()
        .ceil()
        .max(0.0)
        .min(extent.width as f32) as i32;
    let bottom = rect
        .bottom()
        .ceil()
        .max(0.0)
        .min(extent.height as f32) as i32;

    if right <= x || bottom <= y {
        None
    } else {
        Some(VkRect2D {
            offset: VkOffset2D { x, y },
            extent: VkExtent2D {
                width: (right - x) as u32,
                height: (bottom - y) as u32,
            },
        })
    }
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

fn create_compute_pipeline(
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

fn create_graphics_pipeline(
    device: &Device,
    render_pass: VkRenderPass,
    layout: VkPipelineLayout,
    vertex: VkShaderModule,
    fragment: VkShaderModule,
    topology: VkPrimitiveTopology,
) -> Result<VkPipeline> {
    let entry = b"main\0".as_ptr() as *const c_char;
    let stages = [
        VkPipelineShaderStageCreateInfo {
            stage: VK_SHADER_STAGE_VERTEX_BIT,
            module: vertex,
            pName: entry,
            ..Default::default()
        },
        VkPipelineShaderStageCreateInfo {
            stage: VK_SHADER_STAGE_FRAGMENT_BIT,
            module: fragment,
            pName: entry,
            ..Default::default()
        },
    ];

    let binding = VkVertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<[f32; 2]>() as u32,
        inputRate: VK_VERTEX_INPUT_RATE_VERTEX,
    };
    let attribute = VkVertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: VK_FORMAT_R32G32_SFLOAT,
        offset: 0,
    };
    let vertex_input = VkPipelineVertexInputStateCreateInfo {
        vertexBindingDescriptionCount: 1,
        pVertexBindingDescriptions: &binding,
        vertexAttributeDescriptionCount: 1,
        pVertexAttributeDescriptions: &attribute,
        ..Default::default()
    };
    let assembly = VkPipelineInputAssemblyStateCreateInfo {
        topology,
        primitiveRestartEnable: VK_FALSE,
        ..Default::default()
    };
    let viewport = VkPipelineViewportStateCreateInfo {
        viewportCount: 1,
        scissorCount: 1,
        ..Default::default()
    };
    let raster = VkPipelineRasterizationStateCreateInfo {
        polygonMode: VK_POLYGON_MODE_FILL,
        cullMode: VK_CULL_MODE_NONE,
        frontFace: VK_FRONT_FACE_COUNTER_CLOCKWISE,
        lineWidth: 1.0,
        ..Default::default()
    };
    let multisample = VkPipelineMultisampleStateCreateInfo {
        rasterizationSamples: VK_SAMPLE_COUNT_1_BIT,
        ..Default::default()
    };
    let blend_attachment = VkPipelineColorBlendAttachmentState {
        blendEnable: VK_TRUE,
        srcColorBlendFactor: VK_BLEND_FACTOR_SRC_ALPHA,
        dstColorBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        colorBlendOp: VK_BLEND_OP_ADD,
        srcAlphaBlendFactor: VK_BLEND_FACTOR_ONE,
        dstAlphaBlendFactor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        alphaBlendOp: VK_BLEND_OP_ADD,
        colorWriteMask: VK_COLOR_COMPONENT_RGBA,
    };
    let blend = VkPipelineColorBlendStateCreateInfo {
        attachmentCount: 1,
        pAttachments: &blend_attachment,
        ..Default::default()
    };
    let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
    let dynamic = VkPipelineDynamicStateCreateInfo {
        dynamicStateCount: dynamic_states.len() as u32,
        pDynamicStates: dynamic_states.as_ptr(),
        ..Default::default()
    };

    let info = VkGraphicsPipelineCreateInfo {
        stageCount: stages.len() as u32,
        pStages: stages.as_ptr(),
        pVertexInputState: &vertex_input,
        pInputAssemblyState: &assembly,
        pViewportState: &viewport,
        pRasterizationState: &raster,
        pMultisampleState: &multisample,
        pColorBlendState: &blend,
        pDynamicState: &dynamic,
        layout,
        renderPass: render_pass,
        subpass: 0,
        basePipelineIndex: -1,
        ..Default::default()
    };

    let mut pipeline: VkPipeline = VK_NULL_HANDLE;
    check("vkCreateGraphicsPipelines", unsafe {
        (device.fns.create_graphics_pipelines)(
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