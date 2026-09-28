//! Sampled images and their descriptor sets.
//!
//! Every texture owns one descriptor set from a shared pool. Binding a texture
//! is therefore one command and draw commands only carry a compact identifier.
//!
//! Single channel images are viewed as white with red mapped into alpha. This
//! lets the common fragment shader multiply a glyph coverage mask by its vertex
//! colour without a format branch.

use crate::core::{Error, Result};
use crate::render::device::Device;
use crate::render::memory::{Buffer, Location};
use crate::render::vk::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureId(pub u32);

impl TextureId {
    pub const INVALID: TextureId = TextureId(u32::MAX);
}

struct Texture {
    image: VkImage,
    memory: VkDeviceMemory,
    view: VkImageView,
    extent: VkExtent2D,
    bytes_per_pixel: u32,
    descriptor: VkDescriptorSet,
    layout: VkImageLayout,
}

pub struct TextureStore {
    textures: Vec<Texture>,
    pool: VkDescriptorPool,
    layout: VkDescriptorSetLayout,
    sampler_linear: VkSampler,
    sampler_nearest: VkSampler,
    capacity: u32,
}

impl TextureStore {
    pub fn new(
        device: &Device,
        layout: VkDescriptorSetLayout,
        capacity: u32,
    ) -> Result<TextureStore> {
        let capacity = capacity.max(1);
        let size = VkDescriptorPoolSize {
            type_: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount: capacity,
        };
        let info = VkDescriptorPoolCreateInfo {
            maxSets: capacity,
            poolSizeCount: 1,
            pPoolSizes: &size,
            ..Default::default()
        };
        let mut pool: VkDescriptorPool = VK_NULL_HANDLE;
        check("vkCreateDescriptorPool", unsafe {
            (device.fns.create_descriptor_pool)(
                device.handle,
                &info,
                NO_ALLOCATOR,
                &mut pool,
            )
        })?;

        Ok(TextureStore {
            textures: Vec::new(),
            pool,
            layout,
            sampler_linear: create_sampler(device, VK_FILTER_LINEAR)?,
            sampler_nearest: create_sampler(device, VK_FILTER_NEAREST)?,
            capacity,
        })
    }

    pub fn descriptor(&self, id: TextureId) -> Option<VkDescriptorSet> {
        self.textures.get(id.0 as usize).map(|texture| texture.descriptor)
    }

    pub fn create_rgba8(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        data: &[u8],
        nearest: bool,
    ) -> Result<TextureId> {
        self.create(device, width, height, VK_FORMAT_R8G8B8A8_UNORM, 4, data, nearest)
    }

    pub fn create_r8(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        data: &[u8],
        nearest: bool,
    ) -> Result<TextureId> {
        self.create(device, width, height, VK_FORMAT_R8_UNORM, 1, data, nearest)
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: VkFormat,
        bytes_per_pixel: u32,
        data: &[u8],
        nearest: bool,
    ) -> Result<TextureId> {
        if width == 0 || height == 0 {
            return Err(Error::vulkan("texture extent is empty"));
        }
        if self.textures.len() as u32 >= self.capacity {
            return Err(Error::vulkan("texture descriptor pool exhausted"));
        }

        let expected = width as usize * height as usize * bytes_per_pixel as usize;
        if !data.is_empty() && data.len() < expected {
            return Err(Error::vulkan("texture data is shorter than its extent"));
        }

        let image_info = VkImageCreateInfo {
            imageType: VK_IMAGE_TYPE_2D,
            format,
            extent: VkExtent3D { width, height, depth: 1 },
            mipLevels: 1,
            arrayLayers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            ..Default::default()
        };
        let mut image: VkImage = VK_NULL_HANDLE;
        check("vkCreateImage", unsafe {
            (device.fns.create_image)(device.handle, &image_info, NO_ALLOCATOR, &mut image)
        })?;

        let mut requirements = VkMemoryRequirements::default();
        unsafe {
            (device.fns.get_image_memory_requirements)(
                device.handle,
                image,
                &mut requirements,
            );
        }
        let memory_type = device
            .find_memory_type(
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            .ok_or_else(|| Error::vulkan("no device local memory type for image"))?;

        let allocation = VkMemoryAllocateInfo {
            allocationSize: requirements.size,
            memoryTypeIndex: memory_type,
            ..Default::default()
        };
        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        check("vkAllocateMemory", unsafe {
            (device.fns.allocate_memory)(
                device.handle,
                &allocation,
                NO_ALLOCATOR,
                &mut memory,
            )
        })?;
        check("vkBindImageMemory", unsafe {
            (device.fns.bind_image_memory)(device.handle, image, memory, 0)
        })?;

        let components = if format == VK_FORMAT_R8_UNORM {
            VkComponentMapping {
                r: VK_COMPONENT_SWIZZLE_ONE,
                g: VK_COMPONENT_SWIZZLE_ONE,
                b: VK_COMPONENT_SWIZZLE_ONE,
                a: VK_COMPONENT_SWIZZLE_R,
            }
        } else {
            VkComponentMapping::default()
        };
        let view_info = VkImageViewCreateInfo {
            image,
            viewType: VK_IMAGE_VIEW_TYPE_2D,
            format,
            components,
            subresourceRange: VkImageSubresourceRange {
                aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            ..Default::default()
        };
        let mut view: VkImageView = VK_NULL_HANDLE;
        check("vkCreateImageView", unsafe {
            (device.fns.create_image_view)(
                device.handle,
                &view_info,
                NO_ALLOCATOR,
                &mut view,
            )
        })?;

        let allocation = VkDescriptorSetAllocateInfo {
            descriptorPool: self.pool,
            descriptorSetCount: 1,
            pSetLayouts: &self.layout,
            ..Default::default()
        };
        let mut descriptor: VkDescriptorSet = VK_NULL_HANDLE;
        check("vkAllocateDescriptorSets", unsafe {
            (device.fns.allocate_descriptor_sets)(
                device.handle,
                &allocation,
                &mut descriptor,
            )
        })?;

        let sampler = if nearest { self.sampler_nearest } else { self.sampler_linear };
        let image_binding = VkDescriptorImageInfo {
            sampler,
            imageView: view,
            imageLayout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        let write = VkWriteDescriptorSet {
            dstSet: descriptor,
            dstBinding: 0,
            descriptorCount: 1,
            descriptorType: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            pImageInfo: &image_binding,
            ..Default::default()
        };
        unsafe {
            (device.fns.update_descriptor_sets)(
                device.handle,
                1,
                &write,
                0,
                std::ptr::null(),
            );
        }

        let id = TextureId(self.textures.len() as u32);
        self.textures.push(Texture {
            image,
            memory,
            view,
            extent: VkExtent2D { width, height },
            bytes_per_pixel,
            descriptor,
            layout: VK_IMAGE_LAYOUT_UNDEFINED,
        });

        if data.is_empty() {
            self.transition_only(device, id)?;
        } else {
            self.update_region(device, id, 0, 0, width, height, &data[..expected])?;
        }

        crate::log_debug!(
            "render",
            "texture {} created {}x{} {}",
            id.0,
            width,
            height,
            format_name(format)
        );
        Ok(id)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_region(
        &mut self,
        device: &Device,
        id: TextureId,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        let (image, extent, bytes_per_pixel, old_layout) = self
            .info(id)
            .ok_or_else(|| Error::vulkan("texture update names an unknown texture"))?;
        if x + width > extent.width || y + height > extent.height {
            return Err(Error::vulkan("texture update leaves the image"));
        }

        let bytes = width as usize * height as usize * bytes_per_pixel as usize;
        if data.len() < bytes {
            return Err(Error::vulkan("texture update data is too short"));
        }

        let mut staging = Buffer::new(
            device,
            bytes as VkDeviceSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            Location::HostVisible,
        )?;
        staging.write(0, &data[..bytes]);
        staging.flush(device)?;

        let range = color_range();
        let (source_stage, source_access) = if old_layout == VK_IMAGE_LAYOUT_UNDEFINED {
            (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0)
        } else {
            (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        let result = device.one_time_submit(|command_buffer| unsafe {
            let to_copy = VkImageMemoryBarrier {
                srcAccessMask: source_access,
                dstAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout: old_layout,
                newLayout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: range,
                ..Default::default()
            };
            (device.fns.cmd_pipeline_barrier)(
                command_buffer,
                source_stage,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &to_copy,
            );

            let copy = VkBufferImageCopy {
                bufferOffset: 0,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: VkImageSubresourceLayers {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel: 0,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                imageOffset: VkOffset3D { x: x as i32, y: y as i32, z: 0 },
                imageExtent: VkExtent3D { width, height, depth: 1 },
            };
            (device.fns.cmd_copy_buffer_to_image)(
                command_buffer,
                staging.handle,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copy,
            );

            let to_sample = VkImageMemoryBarrier {
                srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
                oldLayout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: range,
                ..Default::default()
            };
            (device.fns.cmd_pipeline_barrier)(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &to_sample,
            );
        });

        staging.destroy(device);
        if result.is_ok() {
            self.note_uploaded(id);
        }
        result
    }

    fn transition_only(&mut self, device: &Device, id: TextureId) -> Result<()> {
        let image = self
            .textures
            .get(id.0 as usize)
            .map(|texture| texture.image)
            .ok_or_else(|| Error::vulkan("texture transition names an unknown texture"))?;
        let range = color_range();

        let result = device.one_time_submit(|command_buffer| unsafe {
            let barrier = VkImageMemoryBarrier {
                srcAccessMask: 0,
                dstAccessMask: VK_ACCESS_SHADER_READ_BIT,
                oldLayout: VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: range,
                ..Default::default()
            };
            (device.fns.cmd_pipeline_barrier)(
                command_buffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &barrier,
            );
        });

        if result.is_ok() {
            self.note_uploaded(id);
        }
        result
    }

    pub fn info(
        &self,
        id: TextureId,
    ) -> Option<(VkImage, VkExtent2D, u32, VkImageLayout)> {
        self.textures.get(id.0 as usize).map(|texture| {
            (
                texture.image,
                texture.extent,
                texture.bytes_per_pixel,
                texture.layout,
            )
        })
    }

    pub fn note_uploaded(&mut self, id: TextureId) {
        if let Some(texture) = self.textures.get_mut(id.0 as usize) {
            texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            for texture in self.textures.drain(..) {
                (device.fns.destroy_image_view)(
                    device.handle,
                    texture.view,
                    NO_ALLOCATOR,
                );
                (device.fns.destroy_image)(
                    device.handle,
                    texture.image,
                    NO_ALLOCATOR,
                );
                (device.fns.free_memory)(
                    device.handle,
                    texture.memory,
                    NO_ALLOCATOR,
                );
            }
            (device.fns.destroy_sampler)(
                device.handle,
                self.sampler_linear,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_sampler)(
                device.handle,
                self.sampler_nearest,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_descriptor_pool)(
                device.handle,
                self.pool,
                NO_ALLOCATOR,
            );
        }
        self.sampler_linear = VK_NULL_HANDLE;
        self.sampler_nearest = VK_NULL_HANDLE;
        self.pool = VK_NULL_HANDLE;
    }
}

fn color_range() -> VkImageSubresourceRange {
    VkImageSubresourceRange {
        aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
        baseMipLevel: 0,
        levelCount: 1,
        baseArrayLayer: 0,
        layerCount: 1,
    }
}

fn create_sampler(device: &Device, filter: VkFilter) -> Result<VkSampler> {
    let info = VkSamplerCreateInfo {
        magFilter: filter,
        minFilter: filter,
        mipmapMode: VK_SAMPLER_MIPMAP_MODE_NEAREST,
        addressModeU: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeV: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        addressModeW: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        minLod: 0.0,
        maxLod: 0.0,
        borderColor: VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        ..Default::default()
    };
    let mut sampler: VkSampler = VK_NULL_HANDLE;
    check("vkCreateSampler", unsafe {
        (device.fns.create_sampler)(device.handle, &info, NO_ALLOCATOR, &mut sampler)
    })?;
    Ok(sampler)
}