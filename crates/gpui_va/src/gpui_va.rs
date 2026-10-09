//! A Linux producer for the surface path: a VA-API decode surface, exported as a dma-buf.
//!
//! The escape hatch lets a producer hand the renderer pixels gpui did not draw, and on Linux the
//! currency is a dma-buf. The shape a video pipeline actually produces — and the one this crate
//! stands in for — is a hardware-decoded `NV12` surface: the VA driver allocates it, a decoder writes
//! it, and `vaExportSurfaceHandle` hands out the object the renderer imports.
//!
//! Two producers make that shape. [`decode`] and [`Decoder`] drive libavcodec over the VA-API device,
//! one exported surface per frame, which is the real thing a video player uses; [`nv12`] hand-fills a
//! surface on the GPU with a flat colour, and stands in where libavcodec is not installed. Both are
//! reached by a client that knows the producer is Linux-only and reaches it behind `#[cfg]`.
//!
//! A decoded surface is **tiled** (here `I915_FORMAT_MOD_Y_TILED`), which the renderer accepts only
//! because its device enables `VK_EXT_image_drm_format_modifier`; see `gpui_wgpu`'s device creation.
//! The surface's memory is not the CPU's to write — iHD maps none of it and refuses `vaPutImage` with
//! `VA_STATUS_ERROR_SURFACE_BUSY` — so the stand-in fills its surface on the GPU, by importing the
//! exported object into a Vulkan device and copying into it, which is what a decoder does with a real
//! bitstream.
//!
//! The decoder also reads each stream's colour space and declares it on the exported handle, so the
//! renderer converts the bytes with the stream's own matrix and range rather than one assumed.
//!
//! `libva` and `libavcodec` are opened at run time, so a machine without them builds and skips.

#![cfg(target_os = "linux")]

use std::collections::VecDeque;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpui_engine::DmaBufHandle;

/// A VA-API surface, kept alive for as long as the renderer samples its dma-buf.
pub struct Surface {
    _va: va::Producer,
}

/// Produce a tiled `NV12` surface filled with `colour`, and the handle the renderer consumes.
///
/// `None` when `libva`, its driver, or the device is unavailable, or when the driver exports a linear
/// surface rather than the tiled one this stands in for. The colour is RGB; the surface carries the
/// Y, Cb and Cr the renderer's BT.601 conversion maps back to it.
pub fn nv12(width: u32, height: u32, colour: [u8; 4]) -> Option<(Surface, DmaBufHandle)> {
    let (producer, handle) = va::produce(width, height)?;
    if handle.modifier == DmaBufHandle::LINEAR {
        log::warn!("gpui_va: the driver exported a linear surface, not the tiled one expected");
        return None;
    }
    if let Err(error) = fill_tiled(&handle, colour) {
        log::warn!("gpui_va: could not fill the surface on the GPU: {error:#}");
        return None;
    }
    Some((Surface { _va: producer }, handle))
}

/// `colour` as the Y, Cb and Cr bytes the renderer's BT.601 conversion maps back to it — the inverse
/// of the shader's matrix, as `gpui_wgpu`'s surface tests also compute it.
fn nv12_from_rgb(rgb: [u8; 4]) -> (u8, u8, u8) {
    let r = f32::from(rgb[0]) / 255.0;
    let g = f32::from(rgb[1]) / 255.0;
    let b = f32::from(rgb[2]) / 255.0;
    let (r, g, b) = (r + 0.7010, g - 0.5291, b + 0.8860);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
    (
        (y.clamp(0.0, 1.0) * 255.0).round() as u8,
        (cb.clamp(0.0, 1.0) * 255.0).round() as u8,
        (cr.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

/// The fixture the demo decodes: a 128×128 Main-profile H.264 keyframe carrying the same SMPTE bars
/// the other tiles do, pinned so the demo and the tests decode one fixed bitstream.
///
/// It is encoded full-range, each bar's `Y`, `Cb` and `Cr` the inverse of the shader's matrix, so the
/// decode and the shader round-trip to the bar colours; a limited-range or differently-matrized
/// encode would instead arrive as washed-out bars.
pub const FIXTURE: &[u8] = include_bytes!("../fixtures/testcard.h264");

/// The player demo's clip: the same bars, 60 frames at 256×144 (about two seconds), scrolled a few
/// pixels per frame so the frames differ and the stream carries real inter-frame (`P`) frames. It is
/// small enough to carry in-tree and decoded the same way a real stream is.
pub const CLIP: &[u8] = include_bytes!("../fixtures/clip.h264");

/// The same one-frame card, encoded **limited** range (BT.601, tagged `MPEG`): the shape almost every
/// real stream is in, and the one that proves the renderer honours the colour space the decoder reads
/// off the stream rather than assuming full range.
pub const FIXTURE_LIMITED: &[u8] = include_bytes!("../fixtures/testcard_limited.h264");

/// A plain-baseline H.264 stream: the bars encoded Baseline with no constraint flag set — the shape an
/// encoder that omits `constraint_set1_flag` leaves, and the one ffmpeg's VA-API hwaccel has no exact
/// profile mapping for. Decoding it proves the decoder opts into a profile mismatch, so the driver
/// takes a stream it would otherwise refuse and decode in software.
pub const BASELINE: &[u8] = include_bytes!("../fixtures/baseline.h264");

/// Decode the first frame of `bitstream` on the GPU and export its surface for the renderer.
///
/// `None` when libavcodec of the declared ABI is not installed, or the stream does not decode: the
/// surface path then simply has nothing to composite. A player drives [`Decoder`] directly instead.
pub fn decode(bitstream: &[u8]) -> Option<Frame> {
    codec::decode(bitstream)
}

pub use codec::{Decoder, Demuxer, Frame};

/// The producer seam: a playback hands the surface path each decoded frame as its period comes due.
impl gpui_surface_producer::SurfaceProducer for Playback {
    fn surface(&mut self) -> Option<gpui_engine::SurfaceSource> {
        self.next().map(Into::into)
    }
}

/// A bounded playback over a stream: it decodes a few frames ahead, hands one out at a time, and
/// recycles a surface only once the consumer says it has finished with it.
///
/// The bound is the point. A VA surface returns to the decoder's pool when its [`Frame`] is dropped,
/// so a player that held every frame would pin every surface it ever decoded, and one that dropped
/// frames too eagerly would let the GPU sample a surface the decoder had already overwritten. This
/// keeps at most the `depth` given to [`open`](Self::open) frames alive, tags every handle it hands
/// out with a release descriptor (see [`DmaBufHandle::release`](gpui_engine::DmaBufHandle::release)),
/// and recycles a frame only after the consumer has signalled it — and never the frame that is still
/// on screen, which a scene re-samples every frame it is held for.
///
/// A playback reads either an elementary stream in memory ([`open`](Self::open)) or a file — a raw
/// stream or a container such as MP4 or Matroska — through libavformat
/// ([`open_path`](Self::open_path)). Either way it loops: when the source drains, a fresh decoder
/// takes over.
pub struct Playback {
    source: Source,
    decoder: Decoder,
    /// Decoded frames not yet handed out.
    queued: VecDeque<Frame>,
    /// Handed out and not yet recycled, oldest first.
    live: VecDeque<Live>,
    /// The handle last handed out, returned again while the consumer has not caught up.
    current: Option<DmaBufHandle>,
    /// The most frames that may be alive at once.
    depth: usize,
    /// How many frames have been handed out, across loops.
    position: u64,
}

/// Where a playback's packets come from.
enum Source {
    /// An elementary stream in memory, fed a chunk at a time; the decoder's own parser frames it.
    Stream {
        bytes: Vec<u8>,
        /// How much of `bytes` has been fed to the decoder.
        fed: usize,
        /// Whether the decoder has been told the stream has ended for this pass.
        ended: bool,
    },
    /// A file read through libavformat, which frames its packets itself.
    Container {
        demuxer: Demuxer,
        /// The file's path, so the stream can be re-opened when it loops.
        path: PathBuf,
        /// Whether the decoder has been flushed for this pass.
        ended: bool,
    },
}

/// A frame handed to the consumer, held so its surface stays allocated until it is recycled.
struct Live {
    _frame: Frame,
    release: Arc<OwnedFd>,
}

/// How much of the stream is fed to the decoder at a time. Small, so one feed decodes only a frame or
/// two — the bound is on *frames*, and a big feed would decode past it.
const FEED_CHUNK: usize = 512;

/// The fewest frames a playback may keep alive: one on screen, one to hand out next.
const MIN_DEPTH: usize = 2;

/// What one feed step did.
enum Fed {
    /// The decoder was handed something.
    Worked,
    /// The source had nothing to give right now; stop until asked again.
    Stalled,
    /// The source is exhausted; start it over so the playback loops.
    Drained,
}

impl Playback {
    /// Open a playback over an elementary stream in memory, keeping at most `depth` frames alive (at
    /// least two), or `None` where the decoder is unavailable or `stream` yields no frame at all.
    pub fn open(stream: &[u8], depth: usize) -> Option<Self> {
        let decoder = Decoder::open()?;
        let mut playback = Self::new(
            Source::Stream {
                bytes: stream.to_vec(),
                fed: 0,
                ended: false,
            },
            decoder,
            depth,
        );
        playback.fill();
        (!playback.queued.is_empty()).then_some(playback)
    }

    /// Open a playback over the file at `path`, decoded through libavformat — a raw elementary stream
    /// or a container — or `None` where libavformat is unavailable, the file cannot be opened, or it
    /// yields no frame at all.
    pub fn open_path(path: &Path, depth: usize) -> Option<Self> {
        let demuxer = Demuxer::open(path)?;
        let decoder = demuxer.decoder()?;
        let mut playback = Self::new(
            Source::Container {
                demuxer,
                path: path.to_owned(),
                ended: false,
            },
            decoder,
            depth,
        );
        playback.fill();
        (!playback.queued.is_empty()).then_some(playback)
    }

    fn new(source: Source, decoder: Decoder, depth: usize) -> Self {
        Self {
            source,
            decoder,
            queued: VecDeque::new(),
            live: VecDeque::new(),
            current: None,
            depth: depth.max(MIN_DEPTH),
            position: 0,
        }
    }

    /// The frame period the source declares, where it declares one, so a player can pace playback at
    /// the stream's own rate. A raw in-memory stream carries no rate; a file is taken at the rate
    /// libavformat reads.
    pub fn frame_period(&self) -> Option<Duration> {
        match &self.source {
            Source::Stream { .. } => None,
            Source::Container { demuxer, .. } => demuxer.frame_period(),
        }
    }

    /// The next frame to show, or the one already showing while the consumer has not released it.
    /// `None` only when the stream yields nothing at all.
    ///
    /// Every handle this hands out is expected to be composited: the renderer signals the handle's
    /// release descriptor once it has sampled the buffer, and only then does the playback return the
    /// surface to the decoder's pool. A consumer that takes a frame and never shows it must signal
    /// that handle's [`release`](gpui_engine::DmaBufHandle::release) itself — a frame nothing ever
    /// samples is a frame nothing ever releases, and the ring waits on it for good.
    pub fn next(&mut self) -> Option<DmaBufHandle> {
        self.recycle();
        if self.live.len() < self.depth {
            if self.queued.is_empty() {
                self.fill();
            }
            if let Some(frame) = self.queued.pop_front() {
                let Some(release) = release_descriptor() else {
                    return self.current.clone();
                };
                let release = Arc::new(release);
                let mut handle = frame.handle().clone();
                handle.release = Some(release.clone());
                self.live.push_back(Live {
                    _frame: frame,
                    release,
                });
                self.current = Some(handle.clone());
                self.position += 1;
                return Some(handle);
            }
        }
        self.current.clone()
    }

    /// How many frames are alive: the ones handed out and not yet recycled.
    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// How many of the live frames the renderer has released — the ones [`live`](Self::live) will
    /// recycle. The rest are still being sampled, or were never composited.
    pub fn signalled(&self) -> usize {
        self.live
            .iter()
            .filter(|live| release_signalled(&live.release))
            .count()
    }

    /// How many frames have been handed out, across loops.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Recycle every released frame except the one still on screen. Keeping the newest alive is what
    /// makes re-showing it (when the consumer has not caught up) safe: the scene may re-sample it at
    /// any time, so its surface must not go back to the pool.
    fn recycle(&mut self) {
        while self.live.len() > 1 {
            let signalled = self
                .live
                .front()
                .is_some_and(|live| release_signalled(&live.release));
            if !signalled {
                break;
            }
            self.live.pop_front();
        }
    }

    /// Feed the source until a frame is ready, it drains, or the budget runs out. The budget bounds
    /// the work (and rules out a spin on a source that decodes nothing).
    fn fill(&mut self) {
        let mut budget = match &self.source {
            Source::Stream { bytes, .. } => 2 * (bytes.len() / FEED_CHUNK + 2),
            Source::Container { .. } => 2 * (self.depth + 8),
        };
        while self.queued.len() < self.depth && budget > 0 {
            budget -= 1;
            let before = self.queued.len();
            match self.feed() {
                Fed::Worked => {}
                Fed::Stalled => return,
                Fed::Drained => self.restart(),
            }
            while self.queued.len() < self.depth {
                match self.decoder.receive() {
                    Some(frame) => self.queued.push_back(frame),
                    None => break,
                }
            }
            if self.queued.len() > before {
                return;
            }
        }
    }

    /// One step of pulling from the source: hand the decoder the next unit, end the source, or report
    /// that it has drained and must start over.
    fn feed(&mut self) -> Fed {
        match &mut self.source {
            Source::Stream { bytes, fed, ended } => {
                if *fed < bytes.len() {
                    let end = (*fed + FEED_CHUNK).min(bytes.len());
                    if self.decoder.send(&bytes[*fed..end]).is_err() {
                        return Fed::Stalled;
                    }
                    *fed = end;
                    Fed::Worked
                } else if !*ended {
                    if self.decoder.finish().is_err() {
                        return Fed::Stalled;
                    }
                    *ended = true;
                    Fed::Worked
                } else if self.queued.is_empty() {
                    Fed::Drained
                } else {
                    Fed::Stalled
                }
            }
            Source::Container { demuxer, ended, .. } => match demuxer.send_next(&mut self.decoder) {
                Ok(true) => Fed::Worked,
                Ok(false) if !*ended => {
                    if self.decoder.finish().is_err() {
                        return Fed::Stalled;
                    }
                    *ended = true;
                    Fed::Worked
                }
                Ok(false) if self.queued.is_empty() => Fed::Drained,
                Ok(false) => Fed::Stalled,
                Err(_) => Fed::Stalled,
            },
        }
    }

    /// Begin the source again, so a drained playback loops.
    fn restart(&mut self) {
        match &mut self.source {
            Source::Stream { fed, ended, .. } => {
                if let Some(decoder) = Decoder::open() {
                    self.decoder = decoder;
                    *fed = 0;
                    *ended = false;
                }
            }
            Source::Container { demuxer, path, ended } => {
                if let Some(fresh) = Demuxer::open(path.as_path())
                    && let Some(decoder) = fresh.decoder()
                {
                    self.decoder = decoder;
                    *demuxer = fresh;
                    *ended = false;
                }
            }
        }
    }
}

/// A fresh `eventfd` the consumer signals once it has finished with a surface. Non-blocking, so a
/// signal can never stall the consumer, and `CLOEXEC`, so it does not leak into a child.
fn release_descriptor() -> Option<OwnedFd> {
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 {
        log::warn!(
            "gpui_va: cannot create a release descriptor: {}",
            std::io::Error::last_os_error()
        );
        return None;
    }
    // SAFETY: `eventfd` returned a fresh descriptor this process owns.
    Some(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Whether the consumer has signalled `release` — a non-blocking `poll`, the same wait the renderer
/// does on an acquire fence.
fn release_signalled(release: &OwnedFd) -> bool {
    let mut descriptor = libc::pollfd {
        fd: release.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    (unsafe { libc::poll(&mut descriptor, 1, 0) }) > 0
}

/// Fill a tiled `Nv12` dma-buf with `colour`, on the GPU.
///
/// The buffer is imported into a Vulkan device the same way the renderer imports it, under the same
/// modifier, and a host-visible staging copy of the linear `NV12` is uploaded into it with
/// `vkCmdCopyBufferToImage`. (`vkCmdClearColorImage` on the plane aspects did nothing on this
/// driver, so the copy is what uploads.)
fn fill_tiled(handle: &DmaBufHandle, colour: [u8; 4]) -> anyhow::Result<()> {
    use std::os::fd::IntoRawFd;

    use ash::vk;

    let entry = unsafe { ash::Entry::load() }
        .map_err(|error| anyhow::anyhow!("load vulkan: {error}"))?;
    let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info),
            None,
        )
    }?;
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let physical = devices
        .iter()
        .copied()
        .find(|device| {
            let properties = unsafe { instance.get_physical_device_properties(*device) };
            properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        })
        .or_else(|| devices.first().copied())
        .ok_or_else(|| anyhow::anyhow!("no Vulkan device to fill on"))?;

    let priority = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&priority);
    let extensions = [
        vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
        vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME.as_ptr(),
        vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME.as_ptr(),
    ];
    let device = unsafe {
        instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&extensions),
            None,
        )
    }?;
    let queue = unsafe { device.get_device_queue(0, 0) };

    // The image the renderer will also import: one object, two planes under the modifier, laid out
    // exactly as `vaExportSurfaceHandle` reported them.
    let luma = &handle.planes[0];
    let chroma = &handle.planes[1];
    let plane_layouts = [
        vk::SubresourceLayout::default()
            .offset(0)
            .row_pitch(u64::from(luma.stride)),
        vk::SubresourceLayout::default()
            .offset(chroma.offset.saturating_sub(luma.offset))
            .row_pitch(u64::from(chroma.stride)),
    ];
    let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(handle.modifier)
        .plane_layouts(&plane_layouts);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
        .extent(vk::Extent3D {
            width: handle.width,
            height: handle.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut modifier_info);
    let image = unsafe { device.create_image(&image_info, None) }?;
    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let fd = luma.fd.try_clone()?.into_raw_fd();
    let mut memory = None;
    let mut last_error = None;
    for index in 0..32u32 {
        if requirements.memory_type_bits & (1 << index) == 0 {
            continue;
        }
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(luma.offset + requirements.size)
            .memory_type_index(index)
            .push_next(&mut import)
            .push_next(&mut dedicated);
        match unsafe { device.allocate_memory(&allocate_info, None) } {
            Ok(allocated) => {
                memory = Some(allocated);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let memory = memory.ok_or_else(|| {
        anyhow::anyhow!("no memory type accepted the fill import (last: {last_error:?})")
    })?;
    unsafe { device.bind_image_memory(image, memory, luma.offset) }?;

    // A host-visible staging buffer holding the linear NV12 to upload; the driver tiles it into the
    // surface, which is how a decoder's output would otherwise arrive.
    let luma_size = u64::from(handle.width) * u64::from(handle.height);
    let chroma_size = u64::from(handle.width / 2) * u64::from(handle.height / 2) * 2;
    let staging = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(luma_size + chroma_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }?;
    let staging_requirements = unsafe { device.get_buffer_memory_requirements(staging) };
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let host_type = (0..32u32)
        .find(|index| {
            let properties = memory_properties.memory_types[*index as usize].property_flags;
            staging_requirements.memory_type_bits & (1 << index) != 0
                && properties.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .ok_or_else(|| anyhow::anyhow!("no host-visible memory for the staging buffer"))?;
    let staging_memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(staging_requirements.size)
                .memory_type_index(host_type),
            None,
        )
    }?;
    unsafe { device.bind_buffer_memory(staging, staging_memory, 0) }?;
    let (y, cb, cr) = nv12_from_rgb(colour);
    unsafe {
        let mapped = device.map_memory(
            staging_memory,
            0,
            staging_requirements.size,
            vk::MemoryMapFlags::empty(),
        )? as *mut u8;
        let bytes = std::slice::from_raw_parts_mut(mapped, (luma_size + chroma_size) as usize);
        bytes[..luma_size as usize].fill(y);
        for pair in bytes[luma_size as usize..].chunks_exact_mut(2) {
            pair[0] = cb;
            pair[1] = cr;
        }
        device.unmap_memory(staging_memory);
    }

    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(0),
            None,
        )
    }?;
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }?[0];

    let plane = |aspect| vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let planes = [
        plane(vk::ImageAspectFlags::PLANE_0),
        plane(vk::ImageAspectFlags::PLANE_1),
    ];
    let barrier = |range, old, new, src, dst| {
        vk::ImageMemoryBarrier::default()
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .src_access_mask(src)
            .dst_access_mask(dst)
    };
    let to_transfer: Vec<_> = planes
        .iter()
        .map(|range| {
            barrier(
                *range,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            )
        })
        .collect();
    let to_sampled: Vec<_> = planes
        .iter()
        .map(|range| {
            barrier(
                *range,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            )
        })
        .collect();
    let regions = [
        vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(handle.width)
            .buffer_image_height(handle.height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: handle.width,
                height: handle.height,
                depth: 1,
            }),
        vk::BufferImageCopy::default()
            .buffer_offset(luma_size)
            .buffer_row_length(handle.width / 2)
            .buffer_image_height(handle.height / 2)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: handle.width / 2,
                height: handle.height / 2,
                depth: 1,
            }),
    ];
    unsafe {
        device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_transfer,
        );
        device.cmd_copy_buffer_to_image(
            command,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_sampled,
        );
        device.end_command_buffer(command)?;

        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let commands = [command];
        device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.device_wait_idle()?;

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_memory, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

/// The version gate for `libavcodec`/`libavutil`/`libavformat`, and where their symbols are
/// resolved.
///
/// ffmpeg bumps a library's soname exactly when its ABI changes, so `libavcodec.so.62` *is* the ABI
/// version — and the versioned declarations this crate carries are written against it. Loading that
/// soname, then resolving a symbol *by version tag* (`LIBAVCODEC_62`), refuses a library that is
/// present but not the ABI declared for, rather than mis-reading its structs.
///
/// `libavformat` is held to the same gate but is optional: it is what opens a container, while the
/// elementary-stream decode stands without it, so a machine that has only `libavcodec` still plays a
/// raw stream.
mod ffmpeg {
    use std::ffi::{CString, c_char, c_int, c_void};

    /// The `(libavcodec, libavutil, libavformat)` majors the declarations are written against.
    ///
    /// libavcodec 62 / libavutil 60 / libavformat 62 is ffmpeg 8.0.
    const ABI: (u32, u32, u32) = (62, 60, 62);

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlvsym(
            handle: *mut c_void,
            symbol: *const c_char,
            version: *const c_char,
        ) -> *mut c_void;
    }

    /// A loaded, ABI-checked `libavcodec` and `libavutil`, held for as long as the decode uses them,
    /// and `libavformat` where it is installed.
    pub struct Ffmpeg {
        codec: *mut c_void,
        util: *mut c_void,
        /// Null where `libavformat` of the declared ABI is not installed, which turns containers off
        /// while leaving the elementary-stream decode working.
        format: *mut c_void,
    }

    impl Ffmpeg {
        /// A symbol from `libavcodec`, resolved under the declared ABI's version tag.
        pub fn codec_symbol(&self, name: &str) -> Option<*mut c_void> {
            resolve(self.codec, "LIBAVCODEC", ABI.0, name)
        }

        /// A symbol from `libavutil`, resolved under the declared ABI's version tag.
        pub fn util_symbol(&self, name: &str) -> Option<*mut c_void> {
            resolve(self.util, "LIBAVUTIL", ABI.1, name)
        }

        /// A symbol from `libavformat`, or `None` where it is not installed.
        pub fn format_symbol(&self, name: &str) -> Option<*mut c_void> {
            if self.format.is_null() {
                return None;
            }
            resolve(self.format, "LIBAVFORMAT", ABI.2, name)
        }
    }

    /// Resolve `name` in `handle` under `tag`'s ABI version — so a library that is not the declared
    /// ABI yields nothing rather than a symbol whose layout we would read wrong.
    fn resolve(handle: *mut c_void, tag: &str, major: u32, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        let version = CString::new(format!("{tag}_{major}")).ok()?;
        let symbol = unsafe { dlvsym(handle, name.as_ptr(), version.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }

    impl Drop for Ffmpeg {
        fn drop(&mut self) {
            unsafe {
                if !self.format.is_null() {
                    dlclose(self.format);
                }
                dlclose(self.util);
                dlclose(self.codec);
            }
        }
    }

    // The handles are process-global library mappings, which are safe to move and share; the decode
    // calls them from one thread.
    unsafe impl Send for Ffmpeg {}
    unsafe impl Sync for Ffmpeg {}

    /// Load the declared ABI, or `None` when `libavcodec`/`libavutil` of that ABI is not installed.
    /// `libavformat` is loaded too where it is, and silently left out where it is not.
    pub fn load() -> Option<Ffmpeg> {
        let (codec_major, util_major, format_major) = ABI;
        let codec = format!("libavcodec.so.{codec_major}");
        let util = format!("libavutil.so.{util_major}");
        // One representative symbol per library, by version tag: this is the ABI assertion.
        if !probe(&codec, "avcodec_find_decoder", &format!("LIBAVCODEC_{codec_major}"))
            || !probe(
                &util,
                "av_hwdevice_ctx_create",
                &format!("LIBAVUTIL_{util_major}"),
            )
        {
            return None;
        }
        let codec = unsafe { dlopen(CString::new(codec).ok()?.as_ptr(), RTLD_NOW) };
        if codec.is_null() {
            return None;
        }
        let util = unsafe { dlopen(CString::new(util).ok()?.as_ptr(), RTLD_NOW) };
        if util.is_null() {
            unsafe { dlclose(codec) };
            return None;
        }
        let format = format!("libavformat.so.{format_major}");
        let format = if probe(
            &format,
            "avformat_open_input",
            &format!("LIBAVFORMAT_{format_major}"),
        ) {
            unsafe { dlopen(CString::new(format).ok()?.as_ptr(), RTLD_NOW) }
        } else {
            std::ptr::null_mut()
        };
        Some(Ffmpeg {
            codec,
            util,
            format,
        })
    }

    /// Whether `soname` exports `symbol` under `version`'s ABI — the check [`load`] gates on.
    pub(crate) fn probe(soname: &str, symbol: &str, version: &str) -> bool {
        let Ok(soname) = CString::new(soname) else {
            return false;
        };
        let Ok(symbol) = CString::new(symbol) else {
            return false;
        };
        let Ok(version) = CString::new(version) else {
            return false;
        };
        let handle = unsafe { dlopen(soname.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return false;
        }
        let resolved = unsafe { dlvsym(handle, symbol.as_ptr(), version.as_ptr()) };
        unsafe { dlclose(handle) };
        !resolved.is_null()
    }

    #[cfg(test)]
    mod tests {
        use super::probe;

        #[test]
        fn the_gate_takes_the_declared_abi_and_refuses_another() {
            assert!(probe(
                "libavcodec.so.62",
                "avcodec_find_decoder",
                "LIBAVCODEC_62"
            ));
            assert!(!probe(
                "libavcodec.so.62",
                "avcodec_find_decoder",
                "LIBAVCODEC_61"
            ));
            assert!(probe(
                "libavutil.so.60",
                "av_hwdevice_ctx_create",
                "LIBAVUTIL_60"
            ));
            assert!(!probe(
                "libavutil.so.60",
                "av_hwdevice_ctx_create",
                "LIBAVUTIL_59"
            ));
            assert!(!probe(
                "libavcodec.so.99",
                "avcodec_find_decoder",
                "LIBAVCODEC_99"
            ));
        }

        /// `libavformat` is held to the same version gate, under its own tag.
        #[test]
        fn the_gate_takes_libavformat_by_its_own_tag() {
            assert!(probe(
                "libavformat.so.62",
                "avformat_open_input",
                "LIBAVFORMAT_62"
            ));
            assert!(!probe(
                "libavformat.so.62",
                "avformat_open_input",
                "LIBAVFORMAT_61"
            ));
        }
    }
}

/// The libavcodec decode: a bitstream in, a VA surface per frame out, gated on the ABI the
/// declarations target.
///
/// A decoder writes into a hardware surface the VA driver allocates, and `frame->data[3]` carries
/// that surface's id; the surface is then exported by the same path the stand-in uses, so the tiled
/// `Nv12` object reaches the renderer identically. The elementary stream is split into access units
/// with `av_parser_parse2` — a whole stream in one `avcodec_send_packet` decodes only its first frame
/// — and each frame the decoder yields carries its surface until the [`Frame`] is dropped. Only
/// `AVCodecContext::hw_device_ctx` is touched, and the offset asserts below are the layout `clang`
/// reports for ffmpeg 8.0.
mod codec {
    #![allow(unsafe_op_in_unsafe_fn)]

    use std::collections::VecDeque;
    use std::ffi::{c_char, c_int, c_void};
    use std::path::Path;
    use std::ptr;
    use std::sync::Arc;
    use std::time::Duration;

    use gpui_engine::DmaBufHandle;

    use super::ffmpeg::Ffmpeg;
    use super::va;

    const AV_HWDEVICE_TYPE_VAAPI: c_int = 3;
    /// `AV_CODEC_ID_H264`, a stable enum value in libavcodec.
    const AV_CODEC_ID_H264: c_int = 27;
    /// `AV_PIX_FMT_VAAPI`: the pixel format a hardware-decoded frame carries — as against the
    /// software format the driver silently falls back to when it will not take a stream.
    const AV_PIX_FMT_VAAPI: c_int = 44;
    /// `AV_HWACCEL_FLAG_ALLOW_PROFILE_MISMATCH`: the opt-in that lets a hwaccel use the closest
    /// profile it supports when a stream names one it has no mapping for.
    const AV_HWACCEL_FLAG_ALLOW_PROFILE_MISMATCH: c_int = 1 << 2;
    /// `AVMEDIA_TYPE_VIDEO`, a stable enum value in libavcodec.
    const AVMEDIA_TYPE_VIDEO: c_int = 0;
    /// `AV_NOPTS_VALUE`: the parser needs a timestamp, and the streams here carry none.
    const AV_NOPTS_VALUE: i64 = i64::MIN;

    /// The colour space the decoder read from a frame's `colorspace`, `color_range` and `height`,
    /// mapped onto the engine's declaration.
    ///
    /// A stream that names no matrix is resolved by height, by the usual convention: standard
    /// definition is BT.601, high definition BT.709. A stream that names no range is taken as
    /// limited — the studio range real streams are almost always in, and the range a full-range
    /// buffer read as limited only loses contrast over, where the reverse clips.
    fn color_space(colorspace: c_int, color_range: c_int, height: c_int) -> gpui_engine::YuvColorSpace {
        use gpui_engine::{YuvColorSpace, YuvMatrix, YuvRange};

        // `AVCOL_RANGE_JPEG` is the only range that means full; `MPEG` and `UNSPECIFIED` are limited.
        let range = match color_range {
            2 => YuvRange::Full,
            _ => YuvRange::Limited,
        };
        let matrix = match colorspace {
            1 => YuvMatrix::Bt709,       // AVCOL_SPC_BT709
            5 | 6 => YuvMatrix::Bt601,   // AVCOL_SPC_BT470BG, AVCOL_SPC_SMPTE170M
            9 | 10 => YuvMatrix::Bt2020, // AVCOL_SPC_BT2020_NCL, AVCOL_SPC_BT2020_CL
            // Unspecified, or a matrix with no RGB equivalent: the resolution default.
            _ if height <= 576 => YuvMatrix::Bt601,
            _ => YuvMatrix::Bt709,
        };
        YuvColorSpace { matrix, range }
    }

    #[repr(C)]
    struct AvBufferRef {
        _buffer: *mut c_void,
        data: *mut u8,
        _size: usize,
    }

    #[repr(C)]
    struct AvHwDeviceContext {
        _class: *const c_void,
        _type: c_int,
        hwctx: *mut c_void,
    }

    #[repr(C)]
    struct AvVaapiDeviceContext {
        display: *mut c_void,
        _quirks: u32,
    }

    /// `AVFrame`, of which only the fields the decode reads are named: the plane pointers, the picture
    /// height (to pick a default matrix when the stream names none), the pixel format (to tell a
    /// hardware frame from one the driver decoded in software), and the colour range and matrix the
    /// decoder read from the bitstream. The offsets are the ones `clang` reports for ffmpeg 8.0 and
    /// are asserted below.
    #[repr(C)]
    struct AvFrame {
        data: [*mut u8; 8],
        _to_height: [u8; 104 - 64],
        _width: c_int,
        height: c_int,
        _nb_samples: c_int,
        format: c_int,
        _to_range: [u8; 280 - 120],
        color_range: c_int,
        _color_primaries: c_int,
        _color_trc: c_int,
        colorspace: c_int,
    }

    #[repr(C)]
    struct AvPacket {
        _buffer: *mut c_void,
        _pts: i64,
        _dts: i64,
        data: *mut u8,
        size: c_int,
    }

    /// `AVCodecContext`, of which `hw_device_ctx` and `hwaccel_flags` are reachable — 560 bytes in,
    /// past the fields we do not name. Those offsets, and the other fields read, are the ones `clang`
    /// reports for ffmpeg 8.0; the asserts below fail the build if a declaration drifts from them.
    #[repr(C)]
    struct AvCodecContext {
        _prefix: [u64; 70],
        hw_device_ctx: *mut AvBufferRef,
        hwaccel_flags: c_int,
    }

    /// An `int`/`int` rational, as libavutil spells one (`num`/`den`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct AvRational {
        num: c_int,
        den: c_int,
    }

    /// `AVFormatContext`, of which only the stream table is reachable: the count, and the streams
    /// themselves, which [`Demuxer`] indexes to reach the video stream's parameters. The offsets are
    /// the ones `clang` reports for ffmpeg 8.0 and are asserted below.
    #[repr(C)]
    struct AvFormatContext {
        _to_nb_streams: [u8; 44],
        nb_streams: u32,
        streams: *mut *mut AvStream,
    }

    /// `AVStream`, of which `codecpar` — the parameters the decoder context is opened from — and the
    /// average frame rate, which paces playback, are named. Both offsets are asserted below.
    #[repr(C)]
    struct AvStream {
        _to_codecpar: [u8; 16],
        codecpar: *mut c_void,
        _to_avg_frame_rate: [u8; 88 - 24],
        avg_frame_rate: AvRational,
    }

    const _: () = {
        assert!(core::mem::offset_of!(AvCodecContext, hw_device_ctx) == 560);
        assert!(core::mem::offset_of!(AvCodecContext, hwaccel_flags) == 568);
        assert!(core::mem::offset_of!(AvFrame, data) == 0);
        assert!(core::mem::offset_of!(AvFrame, height) == 108);
        assert!(core::mem::offset_of!(AvFrame, format) == 116);
        assert!(core::mem::offset_of!(AvFrame, color_range) == 280);
        assert!(core::mem::offset_of!(AvFrame, colorspace) == 292);
        assert!(core::mem::offset_of!(AvPacket, data) == 24);
        assert!(core::mem::offset_of!(AvPacket, size) == 32);
        assert!(core::mem::offset_of!(AvBufferRef, data) == 8);
        assert!(core::mem::offset_of!(AvHwDeviceContext, hwctx) == 16);
        assert!(core::mem::offset_of!(AvVaapiDeviceContext, display) == 0);
        assert!(core::mem::offset_of!(AvFormatContext, nb_streams) == 44);
        assert!(core::mem::offset_of!(AvFormatContext, streams) == 48);
        assert!(core::mem::offset_of!(AvStream, codecpar) == 16);
        assert!(core::mem::offset_of!(AvStream, avg_frame_rate) == 88);
    };

    type HwDeviceCtxCreate =
        unsafe extern "C" fn(*mut *mut AvBufferRef, c_int, *const c_char, *mut c_void, c_int) -> c_int;
    type FindDecoderByName = unsafe extern "C" fn(*const c_char) -> *const c_void;
    type AllocContext3 = unsafe extern "C" fn(*const c_void) -> *mut AvCodecContext;
    type Open2 = unsafe extern "C" fn(*mut AvCodecContext, *const c_void, *mut *mut c_void) -> c_int;
    type SendPacket = unsafe extern "C" fn(*mut AvCodecContext, *const AvPacket) -> c_int;
    type ReceiveFrame = unsafe extern "C" fn(*mut AvCodecContext, *mut AvFrame) -> c_int;
    type FreeContext = unsafe extern "C" fn(*mut *mut AvCodecContext);
    type PacketAlloc = unsafe extern "C" fn() -> *mut AvPacket;
    type PacketFree = unsafe extern "C" fn(*mut *mut AvPacket);
    type NewPacket = unsafe extern "C" fn(*mut AvPacket, c_int) -> c_int;
    type FrameAlloc = unsafe extern "C" fn() -> *mut AvFrame;
    type FrameFree = unsafe extern "C" fn(*mut *mut AvFrame);
    type BufferRef = unsafe extern "C" fn(*mut AvBufferRef) -> *mut AvBufferRef;
    type BufferUnref = unsafe extern "C" fn(*mut *mut AvBufferRef);
    type ParserInit = unsafe extern "C" fn(c_int) -> *mut c_void;
    type ParserParse2 = unsafe extern "C" fn(
        *mut c_void,
        *mut AvCodecContext,
        *mut *mut u8,
        *mut c_int,
        *const u8,
        c_int,
        i64,
        i64,
        i64,
    ) -> c_int;
    type ParserClose = unsafe extern "C" fn(*mut c_void);
    type ParametersToContext = unsafe extern "C" fn(*mut AvCodecContext, *const c_void) -> c_int;
    type PacketUnref = unsafe extern "C" fn(*mut AvPacket);

    /// The libavformat functions that open a file, find its video stream and read its packets.
    type OpenInput = unsafe extern "C" fn(
        *mut *mut AvFormatContext,
        *const c_char,
        *const c_void,
        *mut *mut c_void,
    ) -> c_int;
    type FindStreamInfo = unsafe extern "C" fn(*mut AvFormatContext, *mut *mut c_void) -> c_int;
    type FindBestStream = unsafe extern "C" fn(
        *mut AvFormatContext,
        c_int,
        c_int,
        c_int,
        *mut *const c_void,
        c_int,
    ) -> c_int;
    type ReadFrame = unsafe extern "C" fn(*mut AvFormatContext, *mut AvPacket) -> c_int;
    type CloseInput = unsafe extern "C" fn(*mut *mut AvFormatContext);

    /// The libavcodec/libavutil functions the decode calls, each resolved under the ABI version tag.
    #[derive(Clone, Copy)]
    struct Symbols {
        hwdevice_ctx_create: HwDeviceCtxCreate,
        find_decoder_by_name: FindDecoderByName,
        alloc_context3: AllocContext3,
        open2: Open2,
        send_packet: SendPacket,
        receive_frame: ReceiveFrame,
        free_context: FreeContext,
        packet_alloc: PacketAlloc,
        packet_free: PacketFree,
        new_packet: NewPacket,
        frame_alloc: FrameAlloc,
        frame_free: FrameFree,
        buffer_ref: BufferRef,
        buffer_unref: BufferUnref,
        parser_init: ParserInit,
        parser_parse2: ParserParse2,
        parser_close: ParserClose,
        parameters_to_context: ParametersToContext,
    }

    impl Symbols {
        fn resolve(libs: &Ffmpeg) -> Option<Self> {
            fn symbol<T>(libs: &Ffmpeg, from_util: bool, name: &str) -> Option<T> {
                let address = if from_util {
                    libs.util_symbol(name)
                } else {
                    libs.codec_symbol(name)
                }?;
                // SAFETY: `T` is a function-pointer type, the same size as the address.
                Some(unsafe { std::mem::transmute_copy(&address) })
            }

            Some(Self {
                hwdevice_ctx_create: symbol(libs, true, "av_hwdevice_ctx_create")?,
                find_decoder_by_name: symbol(libs, false, "avcodec_find_decoder_by_name")?,
                alloc_context3: symbol(libs, false, "avcodec_alloc_context3")?,
                open2: symbol(libs, false, "avcodec_open2")?,
                send_packet: symbol(libs, false, "avcodec_send_packet")?,
                receive_frame: symbol(libs, false, "avcodec_receive_frame")?,
                free_context: symbol(libs, false, "avcodec_free_context")?,
                packet_alloc: symbol(libs, false, "av_packet_alloc")?,
                packet_free: symbol(libs, false, "av_packet_free")?,
                new_packet: symbol(libs, false, "av_new_packet")?,
                frame_alloc: symbol(libs, true, "av_frame_alloc")?,
                frame_free: symbol(libs, true, "av_frame_free")?,
                buffer_ref: symbol(libs, true, "av_buffer_ref")?,
                buffer_unref: symbol(libs, true, "av_buffer_unref")?,
                parser_init: symbol(libs, false, "av_parser_init")?,
                parser_parse2: symbol(libs, false, "av_parser_parse2")?,
                parser_close: symbol(libs, false, "av_parser_close")?,
                parameters_to_context: symbol(libs, false, "avcodec_parameters_to_context")?,
            })
        }
    }

    /// The libavformat and libavcodec functions the demuxer calls, resolved only where `libavformat`
    /// of the declared ABI is installed — so a machine without it decodes elementary streams and
    /// simply cannot open a container.
    struct FormatSymbols {
        open_input: OpenInput,
        find_stream_info: FindStreamInfo,
        find_best_stream: FindBestStream,
        read_frame: ReadFrame,
        close_input: CloseInput,
        packet_alloc: PacketAlloc,
        packet_free: PacketFree,
        packet_unref: PacketUnref,
    }

    impl FormatSymbols {
        fn resolve(libs: &Ffmpeg) -> Option<Self> {
            fn symbol<T>(address: Option<*mut c_void>) -> Option<T> {
                let address = address?;
                // SAFETY: `T` is a function-pointer type, the same size as the address.
                Some(unsafe { std::mem::transmute_copy(&address) })
            }

            Some(Self {
                open_input: symbol(libs.format_symbol("avformat_open_input"))?,
                find_stream_info: symbol(libs.format_symbol("avformat_find_stream_info"))?,
                find_best_stream: symbol(libs.format_symbol("av_find_best_stream"))?,
                read_frame: symbol(libs.format_symbol("av_read_frame"))?,
                close_input: symbol(libs.format_symbol("avformat_close_input"))?,
                packet_alloc: symbol(libs.codec_symbol("av_packet_alloc"))?,
                packet_free: symbol(libs.codec_symbol("av_packet_free"))?,
                packet_unref: symbol(libs.codec_symbol("av_packet_unref"))?,
            })
        }
    }

    /// A long-lived decode: the libavcodec context, the VA-API device under it, and the parser that
    /// splits the elementary stream into access units, all kept across frames so a video decodes as a
    /// stream rather than one shot.
    pub struct Decoder {
        libs: Arc<Ffmpeg>,
        symbols: Symbols,
        parser: *mut c_void,
        context: *mut AvCodecContext,
        hw_device: *mut AvBufferRef,
        packet: *mut AvPacket,
        /// The VA display `hw_device` owns, which the surface export needs.
        display: *mut c_void,
        /// Frames pulled from the decoder and not yet handed out. Draining after each access unit is
        /// what keeps `avcodec_send_packet` from returning `EAGAIN`.
        queue: VecDeque<Frame>,
        /// Set once a frame arrives in a software pixel format, so the refusal is reported a single
        /// time rather than once per frame.
        not_hardware: bool,
    }

    impl Decoder {
        /// Open a decoder over the VA-API device, or `None` where libavcodec of the declared ABI, or
        /// the device, is unavailable. The stream is framed by the decoder's own parser, so this is
        /// the elementary-stream (Annex B) path; a container opens through [`Demuxer`] instead.
        pub fn open() -> Option<Self> {
            Self::open_with(Arc::new(super::ffmpeg::load()?), ptr::null_mut())
        }

        /// Open a decoder over a container's video stream: `parameters` are the stream's codec
        /// parameters, which carry the parameter sets a container keeps out of band. The libraries
        /// the demuxer loaded are shared rather than loaded again.
        fn open_demuxed(libs: Arc<Ffmpeg>, parameters: *mut c_void) -> Option<Self> {
            Self::open_with(libs, parameters)
        }

        fn open_with(libs: Arc<Ffmpeg>, parameters: *mut c_void) -> Option<Self> {
            let symbols = Symbols::resolve(&libs)?;
            match unsafe { Self::open_inner(libs, symbols, parameters) } {
                Ok(decoder) => Some(decoder),
                Err(error) => {
                    log::warn!("gpui_va: opening the decoder: {error:#}");
                    None
                }
            }
        }

        unsafe fn open_inner(
            libs: Arc<Ffmpeg>,
            symbols: Symbols,
            parameters: *mut c_void,
        ) -> anyhow::Result<Self> {
            let mut decoder = Self {
                libs,
                symbols,
                parser: ptr::null_mut(),
                context: ptr::null_mut(),
                hw_device: ptr::null_mut(),
                packet: ptr::null_mut(),
                display: ptr::null_mut(),
                queue: VecDeque::new(),
                not_hardware: false,
            };
            // From here any `?` drops `decoder`, freeing whatever has been built.
            let symbols = decoder.symbols;
            anyhow::ensure!(
                (symbols.hwdevice_ctx_create)(
                    &mut decoder.hw_device,
                    AV_HWDEVICE_TYPE_VAAPI,
                    c"/dev/dri/renderD128".as_ptr(),
                    ptr::null_mut(),
                    0,
                ) >= 0,
                "av_hwdevice_ctx_create failed"
            );
            anyhow::ensure!(!decoder.hw_device.is_null(), "no hardware device");

            // ffmpeg owns the VA display, and the export needs the same one.
            let device_context = (*decoder.hw_device).data as *mut AvHwDeviceContext;
            anyhow::ensure!(!device_context.is_null(), "the device has no context");
            let vaapi = (*device_context).hwctx as *mut AvVaapiDeviceContext;
            anyhow::ensure!(!vaapi.is_null(), "the device has no VA-API context");
            decoder.display = (*vaapi).display;
            anyhow::ensure!(!decoder.display.is_null(), "the VA-API device has no display");

            let codec = (symbols.find_decoder_by_name)(c"h264".as_ptr());
            anyhow::ensure!(!codec.is_null(), "no h264 decoder");
            decoder.context = (symbols.alloc_context3)(codec);
            anyhow::ensure!(!decoder.context.is_null(), "avcodec_alloc_context3 failed");
            (*decoder.context).hw_device_ctx = (symbols.buffer_ref)(decoder.hw_device);
            // A stream may name a profile the hwaccel has no mapping for — plain H.264 Baseline, for
            // one, which ffmpeg maps only in its constrained form. With this flag it falls back to
            // the closest profile the driver does support, which decodes the stream the same way,
            // rather than refusing and silently decoding it in software (whose frames have no
            // surface to export). See `vaapi_decode_make_config`'s profile-match handling.
            (*decoder.context).hwaccel_flags |= AV_HWACCEL_FLAG_ALLOW_PROFILE_MISMATCH;
            if !parameters.is_null() {
                // A container keeps the parameter sets out of band, in the stream's `extradata`, and
                // its packets are already framed — copying the parameters in is what lets those
                // packets decode, where an elementary stream carries both in its own bytes.
                anyhow::ensure!(
                    (symbols.parameters_to_context)(decoder.context, parameters) >= 0,
                    "avcodec_parameters_to_context failed"
                );
            }
            anyhow::ensure!(
                (symbols.open2)(decoder.context, codec, ptr::null_mut()) >= 0,
                "avcodec_open2 failed"
            );

            if parameters.is_null() {
                decoder.parser = (symbols.parser_init)(AV_CODEC_ID_H264);
                anyhow::ensure!(!decoder.parser.is_null(), "av_parser_init failed");
            }
            decoder.packet = (symbols.packet_alloc)();
            anyhow::ensure!(!decoder.packet.is_null(), "av_packet_alloc failed");
            Ok(decoder)
        }

        /// Feed `chunk` of the elementary stream: its access units are parsed out and sent to the
        /// decoder, ready for [`Decoder::receive`].
        pub fn send(&mut self, chunk: &[u8]) -> anyhow::Result<()> {
            anyhow::ensure!(
                !self.parser.is_null(),
                "this decoder is fed demuxed packets, not an elementary stream"
            );
            let mut data = chunk;
            while !data.is_empty() {
                let (out, out_size, used) = self.parse(data)?;
                data = &data[used as usize..];
                if out_size > 0 {
                    self.send_access_unit(out, out_size)?;
                }
            }
            Ok(())
        }

        /// End the stream: the parser's last access unit is only emitted now, then the decoder is
        /// flushed. After this, drain [`Decoder::receive`] until it yields `None`.
        pub fn finish(&mut self) -> anyhow::Result<()> {
            // A container framed every packet already, so only the decoder itself is flushed.
            if !self.parser.is_null() {
                loop {
                    let (out, out_size, _) = self.parse(&[])?;
                    if out_size <= 0 {
                        break;
                    }
                    self.send_access_unit(out, out_size)?;
                }
            }
            unsafe { (self.symbols.send_packet)(self.context, ptr::null()) };
            self.pump();
            Ok(())
        }

        /// Feed one packet a container framed, ready for [`Decoder::receive`]. The decoder references
        /// the packet, so the demuxer may release it as soon as this returns.
        fn send_packet(&mut self, packet: *mut AvPacket) -> anyhow::Result<()> {
            anyhow::ensure!(
                unsafe { (self.symbols.send_packet)(self.context, packet) } >= 0,
                "avcodec_send_packet failed"
            );
            // Drain what this packet produced, so the next send is accepted.
            self.pump();
            Ok(())
        }

        fn parse(&mut self, data: &[u8]) -> anyhow::Result<(*mut u8, c_int, c_int)> {
            let mut out: *mut u8 = ptr::null_mut();
            let mut out_size: c_int = 0;
            // An empty slice flushes the parser, which is how the last access unit is emitted.
            let (buf, len) = if data.is_empty() {
                (ptr::null(), 0)
            } else {
                (data.as_ptr(), data.len() as c_int)
            };
            let used = unsafe {
                (self.symbols.parser_parse2)(
                    self.parser,
                    self.context,
                    &mut out,
                    &mut out_size,
                    buf,
                    len,
                    AV_NOPTS_VALUE,
                    AV_NOPTS_VALUE,
                    0,
                )
            };
            anyhow::ensure!(used >= 0, "av_parser_parse2 failed");
            Ok((out, out_size, used))
        }

        fn send_access_unit(&mut self, data: *mut u8, size: c_int) -> anyhow::Result<()> {
            unsafe {
                anyhow::ensure!(
                    (self.symbols.new_packet)(self.packet, size) >= 0,
                    "av_new_packet failed"
                );
                ptr::copy_nonoverlapping(data, (*self.packet).data, size as usize);
                anyhow::ensure!(
                    (self.symbols.send_packet)(self.context, self.packet) >= 0,
                    "avcodec_send_packet failed"
                );
            }
            // Drain what this access unit produced, so the next send is accepted.
            self.pump();
            Ok(())
        }

        /// The next decoded frame, or `None` once the queue is drained.
        pub fn receive(&mut self) -> Option<Frame> {
            self.queue.pop_front()
        }

        /// Pull every frame the decoder has ready into the queue.
        fn pump(&mut self) {
            while let Some(frame) = self.decode_one() {
                self.queue.push_back(frame);
            }
        }

        /// One `receive_frame`, exported; `None` when the decoder has nothing more right now.
        fn decode_one(&mut self) -> Option<Frame> {
            let frame = unsafe { (self.symbols.frame_alloc)() };
            if frame.is_null() {
                return None;
            }
            let mut frame = frame;
            if unsafe { (self.symbols.receive_frame)(self.context, frame) } < 0 {
                unsafe { (self.symbols.frame_free)(&mut frame) };
                return None;
            }
            let (colorspace, color_range, height, format) = unsafe {
                (
                    (*frame).colorspace,
                    (*frame).color_range,
                    (*frame).height,
                    (*frame).format,
                )
            };
            if format != AV_PIX_FMT_VAAPI {
                // The driver would not hardware-decode the stream and fell back to software, whose
                // frames live in ordinary memory rather than a VA surface — there is nothing to
                // export as a dma-buf, so the frame cannot reach the renderer.
                if !self.not_hardware {
                    self.not_hardware = true;
                    log::warn!(
                        "gpui_va: the driver did not hardware-decode this stream and fell back to \
                         software; software frames cannot be exported as a dma-buf, so there is \
                         nothing to composite."
                    );
                }
                unsafe { (self.symbols.frame_free)(&mut frame) };
                return None;
            }
            let surface = unsafe { (*frame).data[3] as usize as u32 };
            let color_space = color_space(colorspace, color_range, height);
            match va::export(self.display, surface) {
                Ok(handle) => Some(Frame {
                    _libs: self.libs.clone(),
                    symbols: self.symbols,
                    frame,
                    handle: handle.with_color_space(color_space),
                }),
                Err(error) => {
                    log::warn!("gpui_va: exporting a decoded surface: {error:#}");
                    unsafe { (self.symbols.frame_free)(&mut frame) };
                    None
                }
            }
        }
    }

    impl Drop for Decoder {
        fn drop(&mut self) {
            // SAFETY: each pointer is one this decoder made; each free accepts a null pointer.
            unsafe {
                (self.symbols.packet_free)(&mut self.packet);
                (self.symbols.parser_close)(self.parser);
                (self.symbols.free_context)(&mut self.context);
                (self.symbols.buffer_unref)(&mut self.hw_device);
            }
        }
    }

    /// One decoded frame: its exported dma-buf, and the `AVFrame` that pins the VA surface behind it.
    ///
    /// A `Frame` is self-contained — the `AVFrame` references the surface, its pool, the device and
    /// the display — so it may outlive the [`Decoder`] it came from, and the renderer may sample its
    /// dma-buf for as long as the frame is held.
    pub struct Frame {
        _libs: Arc<Ffmpeg>,
        symbols: Symbols,
        frame: *mut AvFrame,
        handle: DmaBufHandle,
    }

    impl Frame {
        /// The dma-buf the renderer imports.
        pub fn handle(&self) -> &DmaBufHandle {
            &self.handle
        }
    }

    impl Drop for Frame {
        fn drop(&mut self) {
            // SAFETY: the frame is one this decoder allocated; the free returns its surface to the pool.
            unsafe { (self.symbols.frame_free)(&mut self.frame) };
        }
    }

    /// A container (or a raw elementary stream in a file) opened through libavformat: it hands the
    /// decoder one framed packet at a time, so a container plays through the same bounded path an
    /// elementary stream does.
    ///
    /// libavformat detects the format, finds the video stream and reads its codec parameters — the
    /// parameter sets a container keeps out of band — which a [`Decoder`] is then opened from. The
    /// demuxer keeps its file open, owns the packet it reads, and releases it once the decoder has
    /// referenced it.
    pub struct Demuxer {
        libs: Arc<Ffmpeg>,
        symbols: FormatSymbols,
        format: *mut AvFormatContext,
        packet: *mut AvPacket,
        /// The video stream's codec parameters, valid for as long as `format` is open.
        parameters: *mut c_void,
        /// The stream's average frame rate, for pacing playback.
        frame_rate: AvRational,
    }

    impl Demuxer {
        /// Open the file at `path` through libavformat, or `None` where libavformat of the declared
        /// ABI is not installed, the file cannot be opened, or it carries no video stream.
        pub fn open(path: &Path) -> Option<Self> {
            let libs = Arc::new(super::ffmpeg::load()?);
            let symbols = FormatSymbols::resolve(&libs)?;
            match unsafe { Self::open_inner(libs, symbols, path) } {
                Ok(demuxer) => Some(demuxer),
                Err(error) => {
                    log::warn!("gpui_va: opening a container: {error:#}");
                    None
                }
            }
        }

        unsafe fn open_inner(
            libs: Arc<Ffmpeg>,
            symbols: FormatSymbols,
            path: &Path,
        ) -> anyhow::Result<Self> {
            let url = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| anyhow::anyhow!("the path is not a valid C string"))?;
            let mut format: *mut AvFormatContext = ptr::null_mut();
            anyhow::ensure!(
                (symbols.open_input)(&mut format, url.as_ptr(), ptr::null(), ptr::null_mut()) >= 0,
                "avformat_open_input failed"
            );
            let mut demuxer = Self {
                libs,
                symbols,
                format,
                packet: ptr::null_mut(),
                parameters: ptr::null_mut(),
                frame_rate: AvRational { num: 0, den: 0 },
            };
            // From here any `?` drops `demuxer`, closing the format.
            anyhow::ensure!(
                (demuxer.symbols.find_stream_info)(demuxer.format, ptr::null_mut()) >= 0,
                "avformat_find_stream_info failed"
            );
            let index = (demuxer.symbols.find_best_stream)(
                demuxer.format,
                AVMEDIA_TYPE_VIDEO,
                -1,
                -1,
                ptr::null_mut(),
                0,
            );
            anyhow::ensure!(index >= 0, "the container carries no video stream");
            let context = &*demuxer.format;
            anyhow::ensure!(
                (index as u32) < context.nb_streams,
                "the video stream index is past the stream table"
            );
            let stream = &**context.streams.add(index as usize);
            anyhow::ensure!(!stream.codecpar.is_null(), "the video stream has no parameters");
            demuxer.parameters = stream.codecpar;
            demuxer.frame_rate = stream.avg_frame_rate;
            demuxer.packet = (demuxer.symbols.packet_alloc)();
            anyhow::ensure!(!demuxer.packet.is_null(), "av_packet_alloc failed");
            Ok(demuxer)
        }

        /// Open a decoder over this container's video stream, on the libraries the demuxer loaded.
        pub fn decoder(&self) -> Option<Decoder> {
            Decoder::open_demuxed(self.libs.clone(), self.parameters)
        }

        /// Read the next packet and hand it to `decoder` — a container frames its packets already, so
        /// they go in ahead of any parser. `Ok(false)` at the end of the stream; an error only where
        /// the decoder refused a packet.
        pub fn send_next(&mut self, decoder: &mut Decoder) -> anyhow::Result<bool> {
            // SAFETY: the packet is one this demuxer allocated; releasing it before re-reading keeps
            // the next read from overwriting a buffer still in use, and it accepts a released packet.
            unsafe { (self.symbols.packet_unref)(self.packet) };
            if unsafe { (self.symbols.read_frame)(self.format, self.packet) } < 0 {
                return Ok(false);
            }
            decoder.send_packet(self.packet)?;
            Ok(true)
        }

        /// The stream's average frame period, where the container declares one, so a player can pace
        /// playback at the rate the stream was authored at.
        pub fn frame_period(&self) -> Option<Duration> {
            let (num, den) = (self.frame_rate.num, self.frame_rate.den);
            (num > 0 && den > 0).then(|| Duration::from_secs_f64(f64::from(den) / f64::from(num)))
        }
    }

    impl Drop for Demuxer {
        fn drop(&mut self) {
            // SAFETY: the packet is one this demuxer allocated; closing the format frees its streams,
            // so the parameters pointer must not outlive this.
            unsafe {
                (self.symbols.packet_free)(&mut self.packet);
                (self.symbols.close_input)(&mut self.format);
            }
        }
    }

    /// Decode the first frame of `bitstream`, one shot: the whole stream is fed and the first frame
    /// returned. The fixture and the surface tests use this; a player drives [`Decoder`] directly.
    pub fn decode(bitstream: &[u8]) -> Option<Frame> {
        let mut decoder = Decoder::open()?;
        if let Err(error) = decoder.send(bitstream).and_then(|()| decoder.finish()) {
            log::warn!("gpui_va: decode failed: {error:#}");
            return None;
        }
        decoder.receive()
    }

    #[cfg(test)]
    mod tests {
        use super::color_space;
        use gpui_engine::{YuvColorSpace, YuvMatrix, YuvRange};

        /// A stream that names its matrix and range is taken at its word.
        #[test]
        fn a_named_colour_space_is_kept() {
            // `AVCOL_SPC_BT709`, `AVCOL_RANGE_JPEG`.
            assert_eq!(
                color_space(1, 2, 720),
                YuvColorSpace {
                    matrix: YuvMatrix::Bt709,
                    range: YuvRange::Full,
                },
            );
        }

        /// A stream that names neither is resolved by convention: standard definition is BT.601, high
        /// definition BT.709, and the range is taken as limited — the defaults real footage needs.
        #[test]
        fn an_unnamed_colour_space_is_resolved_by_height() {
            // `AVCOL_SPC_UNSPECIFIED`, `AVCOL_RANGE_UNSPECIFIED`.
            assert_eq!(color_space(2, 0, 576), YuvColorSpace::BT601_LIMITED);
            assert_eq!(color_space(2, 0, 1080), YuvColorSpace::BT709_LIMITED);
        }
    }
}

/// The VA-API FFI: create an NV12 surface with the driver, and export it as a dma-buf.
mod va {
    #![allow(non_upper_case_globals)]
    #![allow(unsafe_op_in_unsafe_fn)]

    use std::ffi::c_void;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    use gpui_engine::{DmaBufHandle, DmaBufPlane, SurfaceFormatKind};
    use libloading::Library;

    // The C layouts, as `va/va.h` and `va/va_drmcommon.h` declare them.
    #[repr(C)]
    struct GenericValue {
        type_: i32,
        _pad: i32,
        value: u64,
    }

    #[repr(C)]
    struct SurfaceAttrib {
        type_: i32,
        flags: u32,
        value: GenericValue,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RmObject {
        fd: i32,
        size: u32,
        drm_format_modifier: u64,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RmLayer {
        drm_format: u32,
        num_planes: u32,
        object_index: [u32; 4],
        offset: [u32; 4],
        pitch: [u32; 4],
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RmDescriptor {
        fourcc: u32,
        width: u32,
        height: u32,
        num_objects: u32,
        objects: [RmObject; 4],
        num_layers: u32,
        layers: [RmLayer; 4],
    }

    const VA_RT_FORMAT_YUV420: u32 = 0x1;
    const VA_FOURCC_NV12: u32 = 0x3231_564E;
    const VASurfaceAttribPixelFormat: i32 = 1;
    const VAGenericValueTypeInteger: i32 = 1;
    const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;
    const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x1;
    const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x4;

    type GetDisplayDrm = unsafe extern "C" fn(i32) -> *mut c_void;
    type Initialize = unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32) -> i32;
    type Terminate = unsafe extern "C" fn(*mut c_void) -> i32;
    type CreateSurfaces = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        *mut u32,
        u32,
        *mut SurfaceAttrib,
        u32,
    ) -> i32;
    type DestroySurfaces = unsafe extern "C" fn(*mut c_void, *mut u32, i32) -> i32;
    type ExportSurfaceHandle =
        unsafe extern "C" fn(*mut c_void, u32, u32, u32, *mut RmDescriptor) -> i32;

    /// The VA display and the exported surface, kept alive while the renderer samples them.
    pub struct Producer {
        _va: Library,
        _drm: Library,
        display: *mut c_void,
        terminate: Terminate,
        destroy_surfaces: DestroySurfaces,
        surface: u32,
    }

    impl Drop for Producer {
        fn drop(&mut self) {
            unsafe {
                (self.destroy_surfaces)(self.display, &mut self.surface, 1);
                (self.terminate)(self.display);
            }
        }
    }

    /// Create an NV12 surface and export it. `None` when `libva`, its driver, or the device is
    /// unavailable; see [`super::nv12`] for the fill.
    pub fn produce(width: u32, height: u32) -> Option<(Producer, DmaBufHandle)> {
        match unsafe { inner(width, height) } {
            Ok(produced) => Some(produced),
            Err(error) => {
                log::warn!("gpui_va: {error:#}");
                None
            }
        }
    }

    unsafe fn inner(width: u32, height: u32) -> anyhow::Result<(Producer, DmaBufHandle)> {
        let va = unsafe { Library::new("libva.so.2") }?;
        let drm = unsafe { Library::new("libva-drm.so.2") }?;
        let get_display: GetDisplayDrm = unsafe { *drm.get(b"vaGetDisplayDRM\0")? };
        let initialize: Initialize = unsafe { *va.get(b"vaInitialize\0")? };
        let terminate: Terminate = unsafe { *va.get(b"vaTerminate\0")? };
        let create_surfaces: CreateSurfaces = unsafe { *va.get(b"vaCreateSurfaces\0")? };
        let destroy_surfaces: DestroySurfaces = unsafe { *va.get(b"vaDestroySurfaces\0")? };

        // The Intel render node, the one the window renderer is steered to as well.
        let node = std::fs::File::open("/dev/dri/renderD128")?;
        let display = get_display(node.as_raw_fd());
        if display.is_null() {
            anyhow::bail!("no VA display on renderD128");
        }
        let mut major = 0;
        let mut minor = 0;
        anyhow::ensure!(
            initialize(display, &mut major, &mut minor) == 0,
            "vaInitialize failed"
        );

        let mut attribs = [SurfaceAttrib {
            type_: VASurfaceAttribPixelFormat,
            flags: 0,
            value: GenericValue {
                type_: VAGenericValueTypeInteger,
                _pad: 0,
                value: u64::from(VA_FOURCC_NV12),
            },
        }];
        let mut surface = 0u32;
        anyhow::ensure!(
            create_surfaces(
                display,
                VA_RT_FORMAT_YUV420,
                width,
                height,
                &mut surface,
                1,
                attribs.as_mut_ptr(),
                attribs.len() as u32,
            ) == 0,
            "vaCreateSurfaces failed"
        );
        // From here on, drop the surface with the display.
        let producer = Producer {
            _va: va,
            _drm: drm,
            display,
            terminate,
            destroy_surfaces,
            surface,
        };

        let handle = export(display, surface)?;
        Ok((producer, handle))
    }

    /// Export `surface`, on `display`, as the dma-buf the renderer consumes.
    ///
    /// The display may be one this module made ([`produce`]) or ffmpeg's (see the crate's decode):
    /// both are the same libva, so the symbol is resolved fresh and the display used as given.
    pub(super) fn export(display: *mut c_void, surface: u32) -> anyhow::Result<DmaBufHandle> {
        let va = unsafe { Library::new("libva.so.2") }?;
        let export: ExportSurfaceHandle = unsafe { *va.get(b"vaExportSurfaceHandle\0")? };
        let mut descriptor = RmDescriptor::default();
        anyhow::ensure!(
            unsafe {
                export(
                    display,
                    surface,
                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                    &mut descriptor,
                )
            } == 0,
            "vaExportSurfaceHandle failed"
        );
        anyhow::ensure!(
            descriptor.num_objects == 1 && descriptor.num_layers == 2,
            "expected one object of two layers, got {} of {}",
            descriptor.num_objects,
            descriptor.num_layers
        );
        let object = descriptor.objects[0];
        let luma = descriptor.layers[0];
        let chroma = descriptor.layers[1];
        let fd = unsafe { OwnedFd::from_raw_fd(object.fd) };
        let chroma_fd = fd.try_clone()?;
        Ok(DmaBufHandle::new(
            descriptor.width,
            descriptor.height,
            SurfaceFormatKind::nv12(),
            object.drm_format_modifier,
            [
                DmaBufPlane::new(fd, u64::from(luma.offset[0]), luma.pitch[0]),
                DmaBufPlane::new(chroma_fd, u64::from(chroma.offset[0]), chroma.pitch[0]),
            ],
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode path, against the checked-in fixture: a real H.264 keyframe decoded by the driver
    /// into a tiled NV12 surface, which is what the renderer composites.
    #[test]
    fn the_fixture_decodes_to_a_tiled_surface() {
        let Some(frame) = decode(FIXTURE) else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };
        let handle = frame.handle();
        assert_eq!((handle.width, handle.height), (128, 128));
        assert_ne!(
            handle.modifier,
            DmaBufHandle::LINEAR,
            "a decoded surface is tiled"
        );
        assert_eq!(handle.plane_count(), 2);
    }

    /// A plain-baseline H.264 stream — profile 66, which ffmpeg's VA-API hwaccel has no exact mapping
    /// for — still decodes in hardware: the decoder opts into a profile mismatch, so it uses the
    /// closest profile the driver supports rather than refusing and falling back to software, where
    /// no surface could be exported. The raw stream and a container both reach a tiled surface.
    #[test]
    fn a_plain_baseline_stream_decodes_in_hardware() {
        let Some(frame) = decode(BASELINE) else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };
        let handle = frame.handle();
        assert_eq!((handle.width, handle.height), (128, 128));
        assert_ne!(
            handle.modifier,
            DmaBufHandle::LINEAR,
            "a hardware-decoded surface is tiled"
        );
        assert_eq!(handle.plane_count(), 2);

        // The same stream behind a file, the route the player takes: a raw elementary stream, whose
        // parameter set rides in the packets, and a container, whose profile comes from its own
        // `avcC` — the case where the hwaccel's refusal once left nothing to composite.
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        for fixture in ["baseline.h264", "baseline.mp4"] {
            let Some(mut playback) = Playback::open_path(&fixtures.join(fixture), 2) else {
                panic!("{fixture} did not decode");
            };
            for _ in 0..3 {
                let handle = playback
                    .next()
                    .unwrap_or_else(|| panic!("a frame from {fixture}"));
                assert_eq!((handle.width, handle.height), (128, 128), "{fixture}");
                signal(&handle);
            }
        }
    }

    /// The decoder reads the stream's colour space off the frame and declares it on the exported
    /// handle: a full-range stream stays full, a limited-range one is declared limited. Without this
    /// a real stream's limited bytes would be read as full range and play washed out.
    #[test]
    fn the_decoded_handle_carries_the_streams_colour_space() {
        use gpui_engine::{YuvColorSpace, YuvMatrix, YuvRange};

        let Some(full) = decode(FIXTURE) else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };
        assert_eq!(
            full.handle().color_space,
            YuvColorSpace {
                matrix: YuvMatrix::Bt601,
                range: YuvRange::Full,
            },
        );

        let Some(limited) = decode(FIXTURE_LIMITED) else {
            eprintln!("skipping: the limited fixture did not decode");
            return;
        };
        assert_eq!(limited.handle().color_space, YuvColorSpace::BT601_LIMITED);
    }

    /// A stream, decoded frame by frame: every frame exports its own tiled surface, and holding them
    /// all — as a player must, so none is recycled while the renderer samples it — keeps them valid.
    #[test]
    fn a_clip_decodes_to_one_tiled_surface_per_frame() {
        let Some(mut decoder) = Decoder::open() else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };
        decoder.send(CLIP).expect("feed the clip");
        decoder.finish().expect("finish the clip");

        let mut frames = Vec::new();
        while let Some(frame) = decoder.receive() {
            let handle = frame.handle();
            assert_eq!((handle.width, handle.height), (256, 144));
            assert_ne!(handle.modifier, DmaBufHandle::LINEAR, "a decoded frame is tiled");
            frames.push(frame);
        }

        assert_eq!(frames.len(), 60, "the clip is 60 frames");
    }

    /// Signal a handle's release descriptor, as the renderer does once it has finished with a frame.
    fn signal(handle: &DmaBufHandle) {
        let Some(release) = handle.release.as_ref() else {
            panic!("a handed-out handle carries a release descriptor");
        };
        let counter: u64 = 1;
        // SAFETY: the descriptor is an `eventfd` the playback made; the pointer and length describe
        // one `u64`, as `eventfd` requires.
        let written = unsafe {
            libc::write(
                release.as_raw_fd(),
                std::ptr::addr_of!(counter).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        assert_eq!(written as usize, std::mem::size_of::<u64>());
    }

    /// However long it plays, a playback keeps only a bounded number of frames alive and cycles the
    /// stream — the point of the ring, where holding every frame would pin every surface.
    #[test]
    fn a_playback_keeps_a_bounded_number_of_frames_alive() {
        let Some(mut playback) = Playback::open(CLIP, 3) else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };

        let mut peak = 0;
        for _ in 0..200 {
            let Some(handle) = playback.next() else {
                break;
            };
            peak = peak.max(playback.live());
            // The consumer is done with the frame at once.
            signal(&handle);
        }

        assert!(peak <= 3, "at most `depth` frames live, saw {peak}");
        assert!(
            playback.position() > 60,
            "the playback looped past the clip's 60 frames, at {}",
            playback.position()
        );
    }

    /// Without a release the playback stops rather than recycle a surface a consumer may still be
    /// sampling; once the outstanding frames are released it advances again.
    #[test]
    fn a_playback_waits_for_the_release_before_recycling() {
        let Some(mut playback) = Playback::open(CLIP, 2) else {
            eprintln!(
                "skipping: libavcodec of the declared ABI, or the VA-API device, is unavailable"
            );
            return;
        };

        // Draw a few frames without releasing any: the ring fills and then stalls.
        let mut drawn = Vec::new();
        for _ in 0..4 {
            drawn.push(playback.next().expect("a frame"));
        }
        assert_eq!(playback.live(), 2, "the bound holds");
        assert_eq!(playback.position(), 2, "and it stops rather than recycle");

        // Release everything the consumer was given, and the playback advances again.
        for handle in &drawn {
            signal(handle);
        }
        let _ = playback.next();
        assert!(
            playback.position() > 2,
            "releasing lets it advance, at {}",
            playback.position()
        );
    }

    /// The checked-in container fixture: the clip muxed into MP4, which frames its packets from the
    /// stream's `extradata` rather than in band, so it decodes only through the demuxer path.
    const FIXTURE_MP4: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/clip.mp4");

    /// libavformat opens the container, finds its video stream, and reads the parameters a decoder is
    /// then opened from — the shape a player needs before it decodes anything.
    #[test]
    fn a_container_demuxes_to_its_video_stream() {
        let Some(demuxer) = Demuxer::open(Path::new(FIXTURE_MP4)) else {
            eprintln!("skipping: libavformat of the declared ABI is unavailable");
            return;
        };
        let period = demuxer
            .frame_period()
            .expect("the container declares a frame rate");
        assert!(
            (period.as_secs_f64() - 0.040).abs() < 1e-3,
            "the clip is 25 fps, saw {period:?}"
        );
    }

    /// A container plays through the same bounded path an elementary stream does: every frame is a
    /// tiled surface carrying the stream's colour space, kept bounded and recycled as the playback
    /// loops.
    #[test]
    fn a_container_plays_through_the_bounded_playback() {
        use gpui_engine::{YuvColorSpace, YuvMatrix, YuvRange};

        let Some(mut playback) = Playback::open_path(Path::new(FIXTURE_MP4), 3) else {
            eprintln!(
                "skipping: libavformat/libavcodec of the declared ABI, or the VA-API device, is \
                 unavailable"
            );
            return;
        };

        let mut peak = 0;
        let mut first = true;
        for _ in 0..200 {
            let Some(handle) = playback.next() else {
                break;
            };
            assert_eq!((handle.width, handle.height), (256, 144));
            assert_ne!(handle.modifier, DmaBufHandle::LINEAR, "a decoded frame is tiled");
            if first {
                first = false;
                // The clip is full-range BT.601, and the colour space is read through the container
                // exactly as it is off a raw stream.
                assert_eq!(
                    handle.color_space,
                    YuvColorSpace {
                        matrix: YuvMatrix::Bt601,
                        range: YuvRange::Full,
                    },
                );
            }
            peak = peak.max(playback.live());
            signal(&handle);
        }

        assert!(peak <= 3, "at most `depth` frames live, saw {peak}");
        assert!(
            playback.position() > 60,
            "the playback looped past the clip's 60 frames, at {}",
            playback.position()
        );
    }
}
