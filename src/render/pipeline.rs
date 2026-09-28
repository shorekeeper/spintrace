//! Graphics pipeline for two dimensional geometry.
//!
//! Positions are framebuffer pixels and the vertex shader maps them directly
//! to clip space through a two-float push constant. One descriptor set carries
//! the texture selected by each draw command.
//!
//! The pipeline consumes position, texture coordinates and normalized colour.
//! The mode word remains in the vertex ABI but is not consumed until a data
//! view requires sampling behaviour that cannot be expressed by image swizzle.

use std::ffi::c_char;

use crate::core::Result;
use crate::render::batch::Vertex;
use crate::render::device::Device;
use crate::render::shaders;
use crate::render::vk::*;

pub const PUSH_BYTES: u32 = 8;

pub struct UiPipeline {
    pub pipeline: VkPipeline,
    pub layout: VkPipelineLayout,
    pub descriptor_layout: VkDescriptorSetLayout,
}

impl UiPipeline {
    pub fn new(device: &Device, render_pass: VkRenderPass) -> Result<UiPipeline> {
        let vertex = create_module(device, &shaders::ui_vertex())?;
        let fragment = create_module(device, &shaders::ui_fragment())?;
        let result = Self::build(device, render_pass, vertex, fragment);

        unsafe {
            (device.fns.destroy_shader_module)(device.handle, vertex, NO_ALLOCATOR);
            (device.fns.destroy_shader_module)(device.handle, fragment, NO_ALLOCATOR);
        }
        result
    }

    fn build(
        device: &Device,
        render_pass: VkRenderPass,
        vertex: VkShaderModule,
        fragment: VkShaderModule,
    ) -> Result<UiPipeline> {
        let binding = VkDescriptorSetLayoutBinding {
            binding: 0,
            descriptorType: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount: 1,
            stageFlags: VK_SHADER_STAGE_FRAGMENT_BIT,
            pImmutableSamplers: std::ptr::null(),
        };
        let descriptor_info = VkDescriptorSetLayoutCreateInfo {
            bindingCount: 1,
            pBindings: &binding,
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

        let binding_description = VkVertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            inputRate: VK_VERTEX_INPUT_RATE_VERTEX,
        };
        let attributes = [
            VkVertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: VK_FORMAT_R32G32_SFLOAT,
                offset: 0,
            },
            VkVertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: VK_FORMAT_R32G32_SFLOAT,
                offset: 8,
            },
            VkVertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: VK_FORMAT_R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        let vertex_input = VkPipelineVertexInputStateCreateInfo {
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: &binding_description,
            vertexAttributeDescriptionCount: attributes.len() as u32,
            pVertexAttributeDescriptions: attributes.as_ptr(),
            ..Default::default()
        };
        let assembly = VkPipelineInputAssemblyStateCreateInfo {
            topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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

        Ok(UiPipeline { pipeline, layout, descriptor_layout })
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            (device.fns.destroy_pipeline)(device.handle, self.pipeline, NO_ALLOCATOR);
            (device.fns.destroy_pipeline_layout)(device.handle, self.layout, NO_ALLOCATOR);
            (device.fns.destroy_descriptor_set_layout)(
                device.handle,
                self.descriptor_layout,
                NO_ALLOCATOR,
            );
        }
        self.pipeline = VK_NULL_HANDLE;
        self.layout = VK_NULL_HANDLE;
        self.descriptor_layout = VK_NULL_HANDLE;
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
        (device.fns.create_shader_module)(device.handle, &info, NO_ALLOCATOR, &mut module)
    })?;
    Ok(module)
}