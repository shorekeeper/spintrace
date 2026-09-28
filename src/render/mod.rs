//! Vulkan renderer facade.
//!
//! Ownership follows the Vulkan object hierarchy. Instance owns the loader and
//! instance dispatch table. Device owns queues and the device dispatch table.
//! Swapchain owns presentation images, views, render pass, framebuffers and
//! completion semaphores.
//!
//! Frame slots own command recording, synchronization and mapped upload buffers.
//! Texture copies and the phase histogram compute pass run before the render
//! pass. Interface base geometry is drawn first, direct spin visualization is
//! drawn second and popup or diagnostic geometry is drawn last.
//!
//! SpinCompute owns asynchronous model execution and readback. SpinView owns
//! frame-local visualization buffers and graphics pipelines. Neither subsystem
//! is exposed through the GUI draw list.

pub mod batch;
pub mod compute;
pub mod compute_shader;
pub mod device;
pub mod frame;
pub mod instance;
pub mod memory;
pub mod pipeline;
pub mod shaders;
pub mod spin_view;
pub mod spin_view_shader;
pub mod surface;
pub mod swapchain;
pub mod texture;
pub mod vk;

use std::ffi::c_void;

use crate::config::settings::{PresentModeCfg, RenderSettings};
use crate::core::{Error, Result};
use crate::platform::Window;
use crate::sim::{SequenceProgram, SimulationConfig, SimulationTrace};

pub use batch::{Color, DrawList, Mode, Rect};
pub use compute::ComputeDiagnostics;
pub use spin_view::SpinVisualFrame;
pub use texture::TextureId;

use compute::SpinCompute;
use spin_view::SpinView;
use device::Device;
use frame::Frame;
use instance::Instance;
use pipeline::{UiPipeline, PUSH_BYTES};
use surface::Surface;
use swapchain::Swapchain;
use texture::TextureStore;
use vk::*;

const STAGING_INITIAL_BYTES: usize = 256 * 1024;

/// Counters from the most recently submitted graphics frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub vertices: u32,
    pub indices: u32,
    pub uploads: u32,
    pub upload_bytes: u32,
    pub spin_draw_calls: u32,
    pub visual_states: u32,
}

struct PendingCopy {
    texture: TextureId,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    offset: usize,
}

/// Top-level owner of Vulkan graphics, compute and presentation resources.
pub struct Renderer {
    instance: Instance,
    device: Device,
    surface: Surface,
    swapchain: Swapchain,
    compute: SpinCompute,
    spin_view: SpinView,
    pipeline: UiPipeline,
    textures: TextureStore,
    frames: Vec<Frame>,
    frame_index: usize,
    white: TextureId,
    size: (u32, u32),
    vsync: bool,
    present_mode: PresentModeCfg,
    background_rgb: u32,
    clear_color: [f32; 4],
    needs_recreate: bool,
    pending: Vec<PendingCopy>,
    upload_scratch: Vec<u8>,
    stats: RenderStats,
}

impl Renderer {
    pub fn new(window: &Window, settings: &RenderSettings) -> Result<Renderer> {
        let instance = Instance::new(settings.validation)?;
        let surface = Surface::new(&instance, window.hwnd(), window.hinstance())?;
        let device = Device::new(&instance, &surface, settings)?;
        let size = window.client_size();

        let swapchain = Swapchain::new(
            &instance,
            &device,
            &surface,
            size,
            settings.vsync,
            settings.present_mode,
            settings.swapchain_images,
            VK_NULL_HANDLE,
        )?;
        if swapchain.render_pass == VK_NULL_HANDLE {
            return Err(Error::vulkan("initial swapchain has no render pass"));
        }

        let pipeline = UiPipeline::new(&device, swapchain.render_pass)?;
        let mut textures =
            TextureStore::new(&device, pipeline.descriptor_layout, settings.max_textures)?;
        let white = textures.create_rgba8(&device, 1, 1, &[255, 255, 255, 255], true)?;
        let compute = SpinCompute::new(&device)?;

        let count = settings.frames_in_flight.max(1);
        let mut frames = Vec::with_capacity(count as usize);
        for slot in 0..count {
            frames.push(Frame::new(
                &device,
                settings.vertex_buffer_kb as usize * 1024,
                settings.index_buffer_kb as usize * 1024,
                STAGING_INITIAL_BYTES,
                slot,
            )?);
        }

        let spin_view =
            SpinView::new(&device, swapchain.render_pass, frames.len())?;
        let clear_color = clear_color(swapchain.format, settings.background_rgb);
        crate::log_info!(
            "render",
            "ready: {} images, {} frames in flight, format {}, present {}",
            swapchain.images.len(),
            frames.len(),
            format_name(swapchain.format),
            present_mode_name(swapchain.present_mode)
        );

        Ok(Renderer {
            instance,
            device,
            surface,
            swapchain,
            compute,
            spin_view,
            pipeline,
            textures,
            frames,
            frame_index: 0,
            white,
            size,
            vsync: settings.vsync,
            present_mode: settings.present_mode,
            background_rgb: settings.background_rgb,
            clear_color,
            needs_recreate: false,
            pending: Vec::with_capacity(16),
            upload_scratch: Vec::with_capacity(STAGING_INITIAL_BYTES),
            stats: RenderStats::default(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device.name
    }

    pub fn white_texture(&self) -> TextureId {
        self.white
    }

    pub fn surface_size(&self) -> (u32, u32) {
        (self.swapchain.extent.width, self.swapchain.extent.height)
    }

    /// Changes the automatic present policy.
    ///
    /// Explicit present modes do not consult this value. It is retained while
    /// one is active and takes effect when the preference returns to Auto.
    pub fn set_vsync(&mut self, enabled: bool) {
        if self.vsync != enabled {
            self.vsync = enabled;
            if self.present_mode == PresentModeCfg::Auto {
                self.needs_recreate = true;
            }
        }
    }

    /// Selects the preferred swapchain presentation mode.
    ///
    /// The surface is queried again during recreation. An unsupported explicit
    /// mode follows the ordinary fallback path in swapchain selection.
    pub fn set_present_mode(&mut self, mode: PresentModeCfg) {
        if self.present_mode != mode {
            self.present_mode = mode;
            self.needs_recreate = true;
        }
    }

    pub fn active_present_mode(&self) -> String {
        vk::present_mode_name(self.swapchain.present_mode)
    }

    pub fn stats(&self) -> RenderStats {
        self.stats
    }

    pub fn compute_diagnostics(&self) -> ComputeDiagnostics {
        self.compute.diagnostics()
    }

    /// Submits a model when the compute queue is idle.
    pub fn request_simulation(
        &mut self,
        config: SimulationConfig,
        program: &SequenceProgram,
        sequence_revision: u64,
    ) -> Result<bool> {
        let Renderer { device, compute, .. } = self;
        compute.request_signal(device, config, program, sequence_revision)
    }

    /// Returns a completed model without waiting for an active one.
    pub fn poll_simulation(&mut self) -> Result<Option<SimulationTrace>> {
        let Renderer { device, compute, .. } = self;
        compute.poll_signal(device)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.size != (width, height) {
            self.size = (width, height);
            self.needs_recreate = true;
        }
    }

    pub fn create_texture_rgba8(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
        nearest: bool,
    ) -> Result<TextureId> {
        self.textures
            .create_rgba8(&self.device, width, height, data, nearest)
    }

    pub fn create_texture_r8(
        &mut self,
        width: u32,
        height: u32,
        data: &[u8],
        nearest: bool,
    ) -> Result<TextureId> {
        self.textures
            .create_r8(&self.device, width, height, data, nearest)
    }

    /// Queues a tightly packed texture region for the next submitted frame.
    pub fn queue_texture_update(
        &mut self,
        texture: TextureId,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        let (_, extent, bytes_per_pixel, _) = self
            .textures
            .info(texture)
            .ok_or_else(|| Error::vulkan("texture update names an unknown texture"))?;
        if x + width > extent.width || y + height > extent.height {
            return Err(Error::vulkan("queued texture update leaves the image"));
        }

        let bytes = width as usize * height as usize * bytes_per_pixel as usize;
        if data.len() < bytes {
            return Err(Error::vulkan("queued texture update data is too short"));
        }

        let offset = (self.upload_scratch.len() + 15) & !15;
        self.upload_scratch.resize(offset, 0);
        self.upload_scratch.extend_from_slice(&data[..bytes]);
        self.pending.push(PendingCopy {
            texture,
            x,
            y,
            width,
            height,
            offset,
        });
        Ok(())
    }

    pub fn render(
        &mut self,
        base: &DrawList,
        top: &DrawList,
        visual: Option<SpinVisualFrame<'_>>,
    ) -> Result<bool> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return Ok(false);
        }
        if self.needs_recreate || self.swapchain.handle == VK_NULL_HANDLE {
            self.recreate_swapchain()?;
            if self.swapchain.handle == VK_NULL_HANDLE {
                return Ok(false);
            }
        }

        let slot = self.frame_index;
        check("vkWaitForFences", unsafe {
            (self.device.fns.wait_for_fences)(
                self.device.handle,
                1,
                &self.frames[slot].in_flight,
                VK_TRUE,
                u64::MAX,
            )
        })?;

        let mut image_index = 0u32;
        let acquired = unsafe {
            (self.device.fns.acquire_next_image_khr)(
                self.device.handle,
                self.swapchain.handle,
                u64::MAX,
                self.frames[slot].image_available,
                VK_NULL_HANDLE,
                &mut image_index,
            )
        };
        match acquired {
            VK_SUCCESS => {}
            VK_SUBOPTIMAL_KHR => self.needs_recreate = true,
            VK_ERROR_OUT_OF_DATE_KHR => {
                self.needs_recreate = true;
                return Ok(false);
            }
            other => return Err(vk_err("vkAcquireNextImageKHR", other)),
        }

        check("vkResetFences", unsafe {
            (self.device.fns.reset_fences)(
                self.device.handle,
                1,
                &self.frames[slot].in_flight,
            )
        })?;

        {
            let Renderer { device, frames, .. } = self;
            frames[slot].upload(device, base, top)?;
        }
        {
            let Renderer { device, frames, upload_scratch, .. } = self;
            frames[slot].upload_staging(device, upload_scratch)?;
        }
        if let Some(frame) = visual {
            self.spin_view.upload(&self.device, slot, frame.states)?;
        }

        self.record(slot, image_index as usize, base, top, visual)?;

        self.stats = RenderStats {
            draw_calls: (base.commands.len() + top.commands.len()) as u32,
            vertices: (base.vertices.len() + top.vertices.len()) as u32,
            indices: (base.indices.len() + top.indices.len()) as u32,
            uploads: self.pending.len() as u32,
            upload_bytes: self.upload_scratch.len() as u32,
            spin_draw_calls: if visual.is_some() {
                1 + visual.and_then(|frame| frame.histogram).is_some() as u32
            } else {
                0
            },
            visual_states: visual.map(|frame| frame.states.len() as u32).unwrap_or(0),
        };

        for copy in self.pending.drain(..) {
            self.textures.note_uploaded(copy.texture);
        }
        self.upload_scratch.clear();

        let waits = [self.frames[slot].image_available];
        let wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT];
        let command_buffers = [self.frames[slot].command_buffer];
        let signals = [self.swapchain.render_finished[image_index as usize]];
        let submit = VkSubmitInfo {
            waitSemaphoreCount: waits.len() as u32,
            pWaitSemaphores: waits.as_ptr(),
            pWaitDstStageMask: wait_stages.as_ptr(),
            commandBufferCount: command_buffers.len() as u32,
            pCommandBuffers: command_buffers.as_ptr(),
            signalSemaphoreCount: signals.len() as u32,
            pSignalSemaphores: signals.as_ptr(),
            ..Default::default()
        };

        check("vkQueueSubmit", unsafe {
            (self.device.fns.queue_submit)(
                self.device.graphics_queue,
                1,
                &submit,
                self.frames[slot].in_flight,
            )
        })?;

        let swapchains = [self.swapchain.handle];
        let indices = [image_index];
        let present = VkPresentInfoKHR {
            waitSemaphoreCount: signals.len() as u32,
            pWaitSemaphores: signals.as_ptr(),
            swapchainCount: swapchains.len() as u32,
            pSwapchains: swapchains.as_ptr(),
            pImageIndices: indices.as_ptr(),
            ..Default::default()
        };
        let presented = unsafe {
            (self.device.fns.queue_present_khr)(self.device.present_queue, &present)
        };
        match presented {
            VK_SUCCESS => {}
            VK_SUBOPTIMAL_KHR | VK_ERROR_OUT_OF_DATE_KHR => self.needs_recreate = true,
            other => return Err(vk_err("vkQueuePresentKHR", other)),
        }

        self.frame_index = (self.frame_index + 1) % self.frames.len();
        Ok(true)
    }

    fn record(
        &self,
        slot: usize,
        image_index: usize,
        base: &DrawList,
        top: &DrawList,
        visual: Option<SpinVisualFrame<'_>>,
    ) -> Result<()> {
        let list = base;
        let frame = &self.frames[slot];
        let command_buffer = frame.command_buffer;
        let extent = self.swapchain.extent;

        check("vkResetCommandPool", unsafe {
            (self.device.fns.reset_command_pool)(
                self.device.handle,
                frame.command_pool,
                0,
            )
        })?;

        let begin = VkCommandBufferBeginInfo {
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ..Default::default()
        };
        check("vkBeginCommandBuffer", unsafe {
            (self.device.fns.begin_command_buffer)(command_buffer, &begin)
        })?;

        unsafe {
            self.record_pending(command_buffer, frame);
            if visual.is_some() {
                self.spin_view
                    .record_prepass(&self.device, command_buffer, slot);
            }

            let clear = VkClearValue { color: self.clear_color };
            let render_pass = VkRenderPassBeginInfo {
                renderPass: self.swapchain.render_pass,
                framebuffer: self.swapchain.framebuffers[image_index],
                renderArea: VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent,
                },
                clearValueCount: 1,
                pClearValues: &clear,
                ..Default::default()
            };
            (self.device.fns.cmd_begin_render_pass)(
                command_buffer,
                &render_pass,
                VK_SUBPASS_CONTENTS_INLINE,
            );
            (self.device.fns.cmd_bind_pipeline)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline.pipeline,
            );

            let viewport = VkViewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                minDepth: 0.0,
                maxDepth: 1.0,
            };
            let scissor = VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent,
            };
            (self.device.fns.cmd_set_viewport)(command_buffer, 0, 1, &viewport);
            (self.device.fns.cmd_set_scissor)(command_buffer, 0, 1, &scissor);

            let push = [extent.width as f32, extent.height as f32];
            (self.device.fns.cmd_push_constants)(
                command_buffer,
                self.pipeline.layout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                PUSH_BYTES,
                push.as_ptr() as *const c_void,
            );

            if !list.indices.is_empty() {
                let vertex_buffers = [frame.vertex.buffer.handle];
                let offsets: [VkDeviceSize; 1] = [0];
                (self.device.fns.cmd_bind_vertex_buffers)(
                    command_buffer,
                    0,
                    1,
                    vertex_buffers.as_ptr(),
                    offsets.as_ptr(),
                );
                (self.device.fns.cmd_bind_index_buffer)(
                    command_buffer,
                    frame.index.buffer.handle,
                    0,
                    VK_INDEX_TYPE_UINT32,
                );

                let mut bound = TextureId::INVALID;
                for command in &list.commands {
                    if command.index_count == 0 {
                        continue;
                    }

                    let x = command.clip[0].max(0).min(extent.width as i32);
                    let y = command.clip[1].max(0).min(extent.height as i32);
                    let width =
                        (command.clip[2].min(extent.width as i32) - x).max(0);
                    let height =
                        (command.clip[3].min(extent.height as i32) - y).max(0);
                    if width == 0 || height == 0 {
                        continue;
                    }

                    let clip = VkRect2D {
                        offset: VkOffset2D { x, y },
                        extent: VkExtent2D {
                            width: width as u32,
                            height: height as u32,
                        },
                    };
                    (self.device.fns.cmd_set_scissor)(
                        command_buffer,
                        0,
                        1,
                        &clip,
                    );

                    if command.texture != bound {
                        let descriptor = self
                            .textures
                            .descriptor(command.texture)
                            .or_else(|| self.textures.descriptor(self.white))
                            .expect("the white texture is missing");
                        (self.device.fns.cmd_bind_descriptor_sets)(
                            command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            self.pipeline.layout,
                            0,
                            1,
                            &descriptor,
                            0,
                            std::ptr::null(),
                        );
                        bound = command.texture;
                    }

                    (self.device.fns.cmd_draw_indexed)(
                        command_buffer,
                        command.index_count,
                        1,
                        command.index_offset,
                        0,
                        0,
                    );
                }
            }

            if let Some(visual) = visual {
                self.spin_view.record_draw(
                    &self.device,
                    command_buffer,
                    slot,
                    visual,
                    extent,
                );
            }

            self.record_top_layer(command_buffer, frame, base, top, extent);
            (self.device.fns.cmd_end_render_pass)(command_buffer);
        }

        check("vkEndCommandBuffer", unsafe {
            (self.device.fns.end_command_buffer)(command_buffer)
        })
    }

    /// Draws the GUI layer that sits above direct Vulkan data views.
    ///
    /// The geometry shares the frame buffers with the base layer and begins
    /// after their byte ranges. Binding offsets keep its local indices valid.
    unsafe fn record_top_layer(
        &self,
        command_buffer: VkCommandBuffer,
        frame: &Frame,
        base: &DrawList,
        top: &DrawList,
        extent: VkExtent2D,
    ) {
        if top.indices.is_empty() {
            return;
        }

        (self.device.fns.cmd_bind_pipeline)(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline.pipeline,
        );

        let viewport = VkViewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            minDepth: 0.0,
            maxDepth: 1.0,
        };
        (self.device.fns.cmd_set_viewport)(command_buffer, 0, 1, &viewport);

        let push = [extent.width as f32, extent.height as f32];
        (self.device.fns.cmd_push_constants)(
            command_buffer,
            self.pipeline.layout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            PUSH_BYTES,
            push.as_ptr() as *const c_void,
        );

        let vertex_offset =
            std::mem::size_of_val(base.vertices.as_slice()) as VkDeviceSize;
        let index_offset =
            std::mem::size_of_val(base.indices.as_slice()) as VkDeviceSize;
        let vertex_buffers = [frame.vertex.buffer.handle];
        let offsets = [vertex_offset];

        (self.device.fns.cmd_bind_vertex_buffers)(
            command_buffer,
            0,
            1,
            vertex_buffers.as_ptr(),
            offsets.as_ptr(),
        );
        (self.device.fns.cmd_bind_index_buffer)(
            command_buffer,
            frame.index.buffer.handle,
            index_offset,
            VK_INDEX_TYPE_UINT32,
        );

        let mut bound = TextureId::INVALID;
        for command in &top.commands {
            if command.index_count == 0 {
                continue;
            }

            let x = command.clip[0].max(0).min(extent.width as i32);
            let y = command.clip[1].max(0).min(extent.height as i32);
            let width = (command.clip[2].min(extent.width as i32) - x).max(0);
            let height = (command.clip[3].min(extent.height as i32) - y).max(0);
            if width == 0 || height == 0 {
                continue;
            }

            let scissor = VkRect2D {
                offset: VkOffset2D { x, y },
                extent: VkExtent2D {
                    width: width as u32,
                    height: height as u32,
                },
            };
            (self.device.fns.cmd_set_scissor)(
                command_buffer,
                0,
                1,
                &scissor,
            );

            if command.texture != bound {
                let descriptor = self
                    .textures
                    .descriptor(command.texture)
                    .or_else(|| self.textures.descriptor(self.white))
                    .expect("the white texture is missing");

                (self.device.fns.cmd_bind_descriptor_sets)(
                    command_buffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.pipeline.layout,
                    0,
                    1,
                    &descriptor,
                    0,
                    std::ptr::null(),
                );
                bound = command.texture;
            }

            (self.device.fns.cmd_draw_indexed)(
                command_buffer,
                command.index_count,
                1,
                command.index_offset,
                0,
                0,
            );
        }
    }

    /// Records every queued texture copy before the first draw that can sample
    /// its destination.
    unsafe fn record_pending(&self, command_buffer: VkCommandBuffer, frame: &Frame) {
        if self.pending.is_empty() {
            return;
        }

        let range = VkImageSubresourceRange {
            aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel: 0,
            levelCount: 1,
            baseArrayLayer: 0,
            layerCount: 1,
        };

        for copy in &self.pending {
            let (image, _, _, layout) = match self.textures.info(copy.texture) {
                Some(info) => info,
                None => continue,
            };
            let (source_stage, source_access) =
                if layout == VK_IMAGE_LAYOUT_UNDEFINED {
                    (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0)
                } else {
                    (
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                    )
                };

            let to_copy = VkImageMemoryBarrier {
                srcAccessMask: source_access,
                dstAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout: layout,
                newLayout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: range,
                ..Default::default()
            };
            (self.device.fns.cmd_pipeline_barrier)(
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

            let region = VkBufferImageCopy {
                bufferOffset: copy.offset as VkDeviceSize,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: VkImageSubresourceLayers {
                    aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel: 0,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                imageOffset: VkOffset3D {
                    x: copy.x as i32,
                    y: copy.y as i32,
                    z: 0,
                },
                imageExtent: VkExtent3D {
                    width: copy.width,
                    height: copy.height,
                    depth: 1,
                },
            };
            (self.device.fns.cmd_copy_buffer_to_image)(
                command_buffer,
                frame.staging.buffer.handle,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &region,
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
            (self.device.fns.cmd_pipeline_barrier)(
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
        }
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return Ok(());
        }

        self.device.wait_idle()?;
        let old = self.swapchain.handle;
        let next = Swapchain::new(
            &self.instance,
            &self.device,
            &self.surface,
            self.size,
            self.vsync,
            self.present_mode,
            0,
            old,
        )?;

        self.swapchain.destroy_dependents(&self.device);
        if old != VK_NULL_HANDLE {
            unsafe {
                (self.device.fns.destroy_swapchain_khr)(
                    self.device.handle,
                    old,
                    NO_ALLOCATOR,
                );
            }
        }

        self.swapchain = next;
        self.clear_color = clear_color(self.swapchain.format, self.background_rgb);
        self.needs_recreate = false;

        crate::log_debug!(
            "render",
            "swapchain rebuilt {}x{} present {}",
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            present_mode_name(self.swapchain.present_mode)
        );
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let _ = self.device.wait_idle();

        for frame in self.frames.drain(..) {
            frame.destroy(&self.device);
        }
        self.compute.destroy(&self.device);
        self.spin_view.destroy(&self.device);
        self.textures.destroy(&self.device);
        self.pipeline.destroy(&self.device);
        self.swapchain.destroy(&self.device);
        self.device.destroy();
        self.surface.destroy(&self.instance);
        self.instance.destroy();

        crate::log_info!("render", "renderer destroyed");
    }
}

fn clear_color(format: VkFormat, rgb: u32) -> [f32; 4] {
    let mut color = [
        ((rgb >> 16) & 0xFF) as f32 / 255.0,
        ((rgb >> 8) & 0xFF) as f32 / 255.0,
        (rgb & 0xFF) as f32 / 255.0,
        1.0,
    ];

    if matches!(format, VK_FORMAT_R8G8B8A8_SRGB | VK_FORMAT_B8G8R8A8_SRGB) {
        for channel in &mut color[..3] {
            *channel = srgb_to_linear(*channel);
        }
    }
    color
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}