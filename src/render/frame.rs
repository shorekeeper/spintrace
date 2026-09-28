//! Per frame command and upload resources.
//!
//! A frame slot owns one command pool, one primary command buffer, an acquire
//! semaphore, a fence and host visible upload buffers. The fence is waited
//! before any buffer in the slot is rewritten.
//!
//! Interface geometry is split into a base layer and a top layer. Vulkan drawn
//! spin vectors and the phase histogram are inserted between them. Popup lists,
//! the diagnostic overlay and the window border therefore remain above the data
//! visualization without requiring another render pass.

use crate::core::Result;
use crate::render::batch::DrawList;
use crate::render::device::Device;
use crate::render::memory::{as_bytes, DynamicBuffer};
use crate::render::vk::*;

pub struct Frame {
    pub command_pool: VkCommandPool,
    pub command_buffer: VkCommandBuffer,
    pub in_flight: VkFence,
    pub image_available: VkSemaphore,
    pub vertex: DynamicBuffer,
    pub index: DynamicBuffer,
    pub staging: DynamicBuffer,
}

impl Frame {
    pub fn new(
        device: &Device,
        vertex_bytes: usize,
        index_bytes: usize,
        staging_bytes: usize,
        slot: u32,
    ) -> Result<Frame> {
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

        let alloc = VkCommandBufferAllocateInfo {
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
            ..Default::default()
        };
        let mut command_buffer: VkCommandBuffer = std::ptr::null_mut();
        check("vkAllocateCommandBuffers", unsafe {
            (device.fns.allocate_command_buffers)(
                device.handle,
                &alloc,
                &mut command_buffer,
            )
        })?;

        let fence_info = VkFenceCreateInfo {
            flags: VK_FENCE_CREATE_SIGNALED_BIT,
            ..Default::default()
        };
        let mut in_flight: VkFence = VK_NULL_HANDLE;
        check("vkCreateFence", unsafe {
            (device.fns.create_fence)(
                device.handle,
                &fence_info,
                NO_ALLOCATOR,
                &mut in_flight,
            )
        })?;

        let semaphore_info = VkSemaphoreCreateInfo::default();
        let mut image_available: VkSemaphore = VK_NULL_HANDLE;
        check("vkCreateSemaphore", unsafe {
            (device.fns.create_semaphore)(
                device.handle,
                &semaphore_info,
                NO_ALLOCATOR,
                &mut image_available,
            )
        })?;

        let vertex =
            DynamicBuffer::new(device, vertex_bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)?;
        let index =
            DynamicBuffer::new(device, index_bytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)?;
        let staging =
            DynamicBuffer::new(device, staging_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)?;

        crate::log_debug!(
            "render",
            "frame {} allocated, vertex {} KB index {} KB staging {} KB",
            slot,
            vertex_bytes / 1024,
            index_bytes / 1024,
            staging_bytes / 1024
        );

        Ok(Frame {
            command_pool,
            command_buffer,
            in_flight,
            image_available,
            vertex,
            index,
            staging,
        })
    }

    /// Uploads both GUI layers into contiguous ranges.
    ///
    /// Each layer is bound with its own byte offsets while recording. Indices
    /// remain relative to the beginning of their own vertex range and require no
    /// rewriting or temporary combined vectors.
    pub fn upload(
        &mut self,
        device: &Device,
        base: &DrawList,
        top: &DrawList,
    ) -> Result<()> {
        let base_vertices = as_bytes(&base.vertices);
        let top_vertices = as_bytes(&top.vertices);
        let base_indices = as_bytes(&base.indices);
        let top_indices = as_bytes(&top.indices);

        self.vertex
            .reserve(device, base_vertices.len() + top_vertices.len())?;
        self.index
            .reserve(device, base_indices.len() + top_indices.len())?;

        if !base_vertices.is_empty() {
            self.vertex.buffer.write(0, base_vertices);
        }
        if !top_vertices.is_empty() {
            self.vertex
                .buffer
                .write(base_vertices.len() as VkDeviceSize, top_vertices);
        }
        if !base_indices.is_empty() {
            self.index.buffer.write(0, base_indices);
        }
        if !top_indices.is_empty() {
            self.index
                .buffer
                .write(base_indices.len() as VkDeviceSize, top_indices);
        }

        self.vertex.buffer.flush(device)?;
        self.index.buffer.flush(device)?;
        Ok(())
    }

    pub fn upload_staging(&mut self, device: &Device, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.staging.reserve(device, bytes.len())?;
        self.staging.write(bytes);
        self.staging.buffer.flush(device)
    }

    pub fn destroy(mut self, device: &Device) {
        unsafe {
            (device.fns.destroy_semaphore)(
                device.handle,
                self.image_available,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_fence)(
                device.handle,
                self.in_flight,
                NO_ALLOCATOR,
            );
            (device.fns.destroy_command_pool)(
                device.handle,
                self.command_pool,
                NO_ALLOCATOR,
            );
        }
        self.vertex.destroy(device);
        self.index.destroy(device);
        self.staging.destroy(device);
    }
}