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
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

pub use codec::{CpuFrame, Decoder, Demuxer, Frame, Readback, ReadbackFrame, StagedReadback};

/// The producer seam: a playback hands the surface path each decoded frame as its period comes due.
impl gpui_video_platform::SurfaceProducer for Playback {
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
    /// While decoding forward to a seek target: that target in the stream's time base. Frames before
    /// it are dropped rather than shown, so a seek lands on the frame asked for, not its keyframe.
    seek_target: Option<i64>,
    /// The newest frame decoded before a seek target, kept so a seek past the last frame shows the
    /// closest frame rather than restarting the stream from the top.
    behind: Option<Frame>,
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
        /// Whether the decoder has been flushed for this pass.
        ended: bool,
    },
}

/// A frame handed to the consumer, held so its surface stays allocated until it is recycled.
struct Live {
    _frame: Frame,
    release: Arc<OwnedFd>,
    /// When the frame was handed out, so a release that never comes can be reclaimed.
    handed: Instant,
}

/// How much of the stream is fed to the decoder at a time. Small, so one feed decodes only a frame or
/// two — the bound is on *frames*, and a big feed would decode past it.
const FEED_CHUNK: usize = 512;

/// The fewest frames a playback may keep alive: one on screen, one to hand out next.
const MIN_DEPTH: usize = 2;

/// How long a handed-out frame waits for its release signal before the playback reclaims it anyway.
///
/// A consumer that never samples a frame — one handed out and then dropped before it reached the
/// screen — never signals it, and the ring would otherwise wait on it for good, freezing that
/// playback. Comfortably longer than any frame's screen life, so a live frame is never reclaimed.
const RELEASE_TIMEOUT: Duration = Duration::from_millis(500);

/// How many packets one fill may decode while seeking forward to a target from the keyframe before it.
/// A seek can cross a whole GOP, so the work is bounded by the file rather than by the queue depth.
const SEEK_BUDGET: usize = 16_384;

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
            seek_target: None,
            behind: None,
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

    /// How long the source is, where it declares a duration: a container knows its length; a raw
    /// in-memory stream does not.
    pub fn duration(&self) -> Option<Duration> {
        match &self.source {
            Source::Stream { .. } => None,
            Source::Container { demuxer, .. } => demuxer.duration(),
        }
    }

    /// Whether [`seek`](Self::seek) lands: a container can be seeked through libavformat; a raw
    /// in-memory elementary stream has no index to seek in.
    pub fn is_seekable(&self) -> bool {
        matches!(self.source, Source::Container { .. })
    }

    /// Move playback to `to`: seek the demuxer, flush the decoder — which invalidates everything it
    /// buffered — drop the frames not yet handed out, and decode forward to the frame at `to`. The
    /// seek itself lands on the keyframe at or before `to`, so the frame asked for is reached by
    /// decoding past the frames between them (see [`fill`](Self::fill)); `false` where the source
    /// cannot seek.
    pub fn seek(&mut self, to: Duration) -> bool {
        let target = match &mut self.source {
            Source::Stream { .. } => return false,
            Source::Container { demuxer, ended, .. } => {
                let Some(target) = demuxer.seek(to) else {
                    return false;
                };
                // The container is no longer at its end: the next read continues from `to`.
                *ended = false;
                target
            }
        };
        self.decoder.flush();
        self.queued.clear();
        self.behind = None;
        self.seek_target = Some(target);
        self.fill();
        true
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
                    handed: Instant::now(),
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
            let front = self.live.front().expect("live.len() > 1");
            // Reclaim a frame the consumer released, or one it never released at all: a frame that
            // was handed out and then dropped before it reached the screen signals nothing, and the
            // ring must not wait on it for good.
            let recyclable = release_signalled(&front.release)
                || front.handed.elapsed() >= RELEASE_TIMEOUT;
            if !recyclable {
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
            // Decoding forward to a seek target can cross a whole GOP, so give it the seek budget;
            // an ordinary fill only needs enough to top the queue up.
            Source::Container { .. } if self.seek_target.is_some() => SEEK_BUDGET,
            Source::Container { .. } => 2 * (self.depth + 8),
        };
        while self.queued.len() < self.depth && budget > 0 {
            budget -= 1;
            let before = self.queued.len();
            match self.feed() {
                Fed::Worked => {}
                Fed::Stalled => return,
                Fed::Drained => {
                    // Seeking past the last frame: show the closest frame before the target rather
                    // than restarting the stream from the top.
                    if let Some(frame) = self.behind.take() {
                        self.seek_target = None;
                        self.queued.push_back(frame);
                        return;
                    }
                    self.restart();
                }
            }
            while self.queued.len() < self.depth {
                match self.decoder.receive() {
                    Some(frame) => self.keep(frame),
                    None => break,
                }
            }
            if self.queued.len() > before {
                return;
            }
        }
    }

    /// Queue a decoded frame. While a seek is being resolved, a frame before the target is dropped —
    /// keeping only the newest of them, so a seek past the last frame still has one to show — and a
    /// frame at or after the target ends the seek, so the frame asked for is the one presented.
    fn keep(&mut self, frame: Frame) {
        if let Some(target) = self.seek_target {
            match frame.pts() {
                Some(pts) if pts < target => {
                    self.behind = Some(frame);
                    return;
                }
                _ => {
                    self.seek_target = None;
                    self.behind = None;
                }
            }
        }
        self.queued.push_back(frame);
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

    /// Begin the source again, so a drained playback loops. The decoder is reused — the parser reset,
    /// the container rewound — rather than reopened: a fresh decoder builds a fresh VA-API device and
    /// surface pool on every loop, which the driver does not hand back, so a wall of looping panes
    /// leaks steadily and ends up memory-bound.
    fn restart(&mut self) {
        match &mut self.source {
            Source::Stream { fed, ended, .. } => {
                self.decoder.rewind();
                *fed = 0;
                *ended = false;
            }
            Source::Container { demuxer, ended } => {
                if demuxer.seek(Duration::ZERO).is_some() {
                    self.decoder.rewind();
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

/// The version gate for `libavcodec`/`libavutil`/`libavformat`/`libswscale`, and where their symbols
/// are resolved.
///
/// ffmpeg bumps a library's soname exactly when its ABI changes, so `libavcodec.so.62` *is* the ABI
/// version — and the versioned declarations this crate carries are written against it. Loading that
/// soname, then resolving a symbol *by version tag* (`LIBAVCODEC_62`), refuses a library that is
/// present but not the ABI declared for, rather than mis-reading its structs.
///
/// `libavformat` and `libswscale` are held to the same gate but are optional: `libavformat` is what
/// opens a container, while the elementary-stream decode stands without it, so a machine that has only
/// `libavcodec` still plays a raw stream; `libswscale` accelerates the readback's software convert,
/// while the scalar converter stands without it, so such a machine reads back slower, not wrongly.
mod ffmpeg {
    use std::ffi::{CString, c_char, c_int, c_void};

    /// The `(libavcodec, libavutil, libavformat)` majors the declarations are written against.
    ///
    /// libavcodec 62 / libavutil 60 / libavformat 62 is ffmpeg 8.0.
    const ABI: (u32, u32, u32) = (62, 60, 62);

    /// The `libswscale` major the declarations are written against (`libswscale.so.9` is ffmpeg 8.0).
    /// Held out of [`ABI`] because the software convert it accelerates is optional: a machine without
    /// it still decodes and still reads back, only through the scalar converter.
    const SWSCALE_ABI: u32 = 9;

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
        /// Null where `libswscale` of the declared ABI is not installed, which turns the readback's
        /// convert down to the scalar path while leaving the decode untouched.
        swscale: *mut c_void,
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

        /// A symbol from `libswscale`, or `None` where it is not installed.
        pub fn swscale_symbol(&self, name: &str) -> Option<*mut c_void> {
            if self.swscale.is_null() {
                return None;
            }
            resolve(self.swscale, "LIBSWSCALE", SWSCALE_ABI, name)
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
                if !self.swscale.is_null() {
                    dlclose(self.swscale);
                }
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
    /// `libavformat` and `libswscale` are loaded too where they are, and silently left out where they
    /// are not.
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
        // libswscale is optional like libavformat: it accelerates the readback's software convert,
        // which the scalar converter otherwise does, so a machine without it is slower, not broken.
        let swscale = format!("libswscale.so.{SWSCALE_ABI}");
        let swscale = if probe(&swscale, "sws_scale", &format!("LIBSWSCALE_{SWSCALE_ABI}")) {
            unsafe { dlopen(CString::new(swscale).ok()?.as_ptr(), RTLD_NOW) }
        } else {
            std::ptr::null_mut()
        };
        Some(Ffmpeg {
            codec,
            util,
            format,
            swscale,
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

        /// `libswscale` is held to the same version gate, under its own tag.
        #[test]
        fn the_gate_takes_libswscale_by_its_own_tag() {
            assert!(probe("libswscale.so.9", "sws_scale", "LIBSWSCALE_9"));
            assert!(!probe("libswscale.so.9", "sws_scale", "LIBSWSCALE_8"));
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
    use std::time::{Duration, Instant};

    use gpui_engine::DmaBufHandle;

    use super::ffmpeg::Ffmpeg;
    use super::va;

    const AV_HWDEVICE_TYPE_VAAPI: c_int = 3;
    /// `AV_CODEC_ID_H264`, a stable enum value in libavcodec.
    const AV_CODEC_ID_H264: c_int = 27;
    /// `AV_PIX_FMT_VAAPI`: the pixel format a hardware-decoded frame carries — as against the
    /// software format the driver silently falls back to when it will not take a stream.
    const AV_PIX_FMT_VAAPI: c_int = 44;
    /// `AV_PIX_FMT_NV12`: the planar 4:2:0 format a readback transfers a hardware frame into, whose
    /// two planes the conversion to packed pixels walks.
    const AV_PIX_FMT_NV12: c_int = 23;
    /// `AV_PIX_FMT_BGRA`: the packed format the readback converts those planes into — the byte order
    /// the renderer samples (`Rgba<u8>` bytes, blue first).
    const AV_PIX_FMT_BGRA: c_int = 28;
    /// `SWS_BILINEAR`: libswscale's scaler, which for a same-size convert is a plain format change.
    const SWS_BILINEAR: c_int = 2;
    /// The `SWS_CS_*` colour-space ids `sws_getCoefficients` takes, one per matrix the engine names.
    const SWS_CS_ITU709: c_int = 1;
    const SWS_CS_ITU601: c_int = 5;
    const SWS_CS_BT2020: c_int = 9;
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

    /// `AVFrame`, of which only the fields the decode and the readback read are named: the plane
    /// pointers and line strides (a readback walks the planes), the picture size, the pixel format (to
    /// tell a hardware frame from one the driver decoded in software), and the colour range and matrix
    /// the decoder read from the bitstream. The offsets are the ones `clang` reports for ffmpeg 8.0 and
    /// are asserted below.
    #[repr(C)]
    struct AvFrame {
        data: [*mut u8; 8],
        /// The byte stride of each plane, so a readback can walk a padded plane.
        linesize: [c_int; 8],
        _to_width: [u8; 104 - 96],
        width: c_int,
        height: c_int,
        _nb_samples: c_int,
        format: c_int,
        // `pict_type` (4) and `sample_aspect_ratio` (8) precede `pts`, which is 8-byte aligned.
        _to_pts: [u8; 136 - 120],
        pts: i64,
        _to_range: [u8; 280 - 144],
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
        /// The index of the stream this packet belongs to, so the demuxer can drop the packets that
        /// are not the video's — a container interleaves its streams, and only one feeds the decoder.
        stream_index: c_int,
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
        // `nb_stream_groups`/`stream_groups` and `nb_chapters`/`chapters` come between `streams` and
        // `url` in the declared ABI (libavformat 62), so `duration` sits past them at offset 104.
        _to_duration: [u8; 104 - 56],
        /// The container's duration, in `AV_TIME_BASE` (microseconds); `AV_NOPTS_VALUE` (negative)
        /// when the format carries none — a raw elementary stream, or a live source.
        duration: i64,
    }

    /// `AVStream`, of which `codecpar` — the parameters the decoder context is opened from — and the
    /// average frame rate, which paces playback, are named. Both offsets are asserted below.
    #[repr(C)]
    struct AvStream {
        _to_codecpar: [u8; 16],
        codecpar: *mut c_void,
        _to_time_base: [u8; 32 - 24],
        /// The stream's time base: what a packet or frame timestamp is measured in.
        time_base: AvRational,
        _to_avg_frame_rate: [u8; 88 - 40],
        avg_frame_rate: AvRational,
    }

    const _: () = {
        assert!(core::mem::offset_of!(AvCodecContext, hw_device_ctx) == 560);
        assert!(core::mem::offset_of!(AvCodecContext, hwaccel_flags) == 568);
        assert!(core::mem::offset_of!(AvFrame, data) == 0);
        assert!(core::mem::offset_of!(AvFrame, linesize) == 64);
        assert!(core::mem::offset_of!(AvFrame, width) == 104);
        assert!(core::mem::offset_of!(AvFrame, height) == 108);
        assert!(core::mem::offset_of!(AvFrame, format) == 116);
        assert!(core::mem::offset_of!(AvFrame, pts) == 136);
        assert!(core::mem::offset_of!(AvFrame, color_range) == 280);
        assert!(core::mem::offset_of!(AvFrame, colorspace) == 292);
        assert!(core::mem::offset_of!(AvPacket, data) == 24);
        assert!(core::mem::offset_of!(AvPacket, size) == 32);
        assert!(core::mem::offset_of!(AvPacket, stream_index) == 36);
        assert!(core::mem::offset_of!(AvBufferRef, data) == 8);
        assert!(core::mem::offset_of!(AvHwDeviceContext, hwctx) == 16);
        assert!(core::mem::offset_of!(AvVaapiDeviceContext, display) == 0);
        assert!(core::mem::offset_of!(AvFormatContext, nb_streams) == 44);
        assert!(core::mem::offset_of!(AvFormatContext, streams) == 48);
        assert!(core::mem::offset_of!(AvFormatContext, duration) == 104);
        assert!(core::mem::offset_of!(AvStream, codecpar) == 16);
        assert!(core::mem::offset_of!(AvStream, time_base) == 32);
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
    /// `av_hwframe_transfer_data`: copies a hardware frame into a software one, which is the GPU→CPU
    /// readback a player that does not import the buffer pays for.
    type HwFrameTransferData =
        unsafe extern "C" fn(*mut AvFrame, *const AvFrame, c_int) -> c_int;
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
    type FlushBuffers = unsafe extern "C" fn(*mut AvCodecContext);

    /// `sws_getContext`: `(srcW, srcH, srcFormat, dstW, dstH, dstFormat, flags, srcFilter, dstFilter,
    /// param)`, returning the convert context or null. The filter and parameter pointers are left null
    /// — for a same-size convert there is nothing to filter.
    type SwsGetContext = unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
        *const c_void,
    ) -> *mut c_void;
    /// `sws_scale`: `(ctx, srcSlice, srcStride, srcSliceY, srcSliceH, dst, dstStride)`, walking the
    /// planes named by the slice arrays and returning the rows written, or negative on error.
    type SwsScale = unsafe extern "C" fn(
        *mut c_void,
        *const *const u8,
        *const c_int,
        c_int,
        c_int,
        *const *mut u8,
        *const c_int,
    ) -> c_int;
    type SwsFreeContext = unsafe extern "C" fn(*mut c_void);
    /// `sws_getCoefficients`: the 16.16 fixed-point YUV↔RGB coefficients for an `SWS_CS_*` id.
    type SwsGetCoefficients = unsafe extern "C" fn(c_int) -> *const c_int;
    /// `sws_setColorspaceDetails`: `(ctx, invTable, srcRange, table, dstRange, brightness, contrast,
    /// saturation)`, so the convert honours the stream's own matrix and range rather than the
    /// default. `invTable`/`table` are the coefficient arrays from [`SwsGetCoefficients`].
    type SwsSetColorspaceDetails = unsafe extern "C" fn(
        *mut c_void,
        *const c_int,
        c_int,
        *const c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int;

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
    /// `av_seek_frame`: `(s, stream_index, timestamp, flags)`, the timestamp in that stream's own
    /// time base.
    type SeekFrame = unsafe extern "C" fn(*mut AvFormatContext, c_int, i64, c_int) -> c_int;

    /// Seek to the keyframe at or before the target, so a seek lands on a frame the decoder can start
    /// from. `AVSEEK_FLAG_BACKWARD`.
    const AVSEEK_FLAG_BACKWARD: c_int = 1;

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
        hwframe_transfer_data: HwFrameTransferData,
        buffer_ref: BufferRef,
        buffer_unref: BufferUnref,
        parser_init: ParserInit,
        parser_parse2: ParserParse2,
        parser_close: ParserClose,
        parameters_to_context: ParametersToContext,
        flush_buffers: FlushBuffers,
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
                hwframe_transfer_data: symbol(libs, true, "av_hwframe_transfer_data")?,
                buffer_ref: symbol(libs, true, "av_buffer_ref")?,
                buffer_unref: symbol(libs, true, "av_buffer_unref")?,
                parser_init: symbol(libs, false, "av_parser_init")?,
                parser_parse2: symbol(libs, false, "av_parser_parse2")?,
                parser_close: symbol(libs, false, "av_parser_close")?,
                parameters_to_context: symbol(libs, false, "avcodec_parameters_to_context")?,
                flush_buffers: symbol(libs, false, "avcodec_flush_buffers")?,
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
        seek_frame: SeekFrame,
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
                seek_frame: symbol(libs.format_symbol("av_seek_frame"))?,
            })
        }
    }

    /// The libswscale functions the readback converts through, resolved only where `libswscale` of
    /// the declared ABI is installed. Optional throughout: where it is not, the scalar converter
    /// stands in, so a readback costs more, but works.
    #[derive(Clone, Copy)]
    struct SwsSymbols {
        get_context: SwsGetContext,
        set_colorspace_details: SwsSetColorspaceDetails,
        get_coefficients: SwsGetCoefficients,
        scale: SwsScale,
        free_context: SwsFreeContext,
    }

    impl SwsSymbols {
        fn resolve(libs: &Ffmpeg) -> Option<Self> {
            // An escape hatch for the benchmark and for diagnosis: with `GPUI_VA_NO_SWSCALE` set, the
            // scalar converter is used even where libswscale is installed, so the two can be compared
            // on one machine.
            if std::env::var_os("GPUI_VA_NO_SWSCALE").is_some() {
                return None;
            }
            fn symbol<T>(address: Option<*mut c_void>) -> Option<T> {
                let address = address?;
                // SAFETY: `T` is a function-pointer type, the same size as the address.
                Some(unsafe { std::mem::transmute_copy(&address) })
            }

            Some(Self {
                get_context: symbol(libs.swscale_symbol("sws_getContext"))?,
                set_colorspace_details: symbol(libs.swscale_symbol("sws_setColorspaceDetails"))?,
                get_coefficients: symbol(libs.swscale_symbol("sws_getCoefficients"))?,
                scale: symbol(libs.swscale_symbol("sws_scale"))?,
                free_context: symbol(libs.swscale_symbol("sws_freeContext"))?,
            })
        }
    }

    /// The libswscale convert context for one source geometry, destination geometry and colour space,
    /// created on first use and recreated when any of them changes. One context is reused across a
    /// stream's frames, which is what keeps the conversion cheap; rebuilding it per frame would
    /// negate the accelerator.
    struct Sws {
        symbols: SwsSymbols,
        context: *mut c_void,
        width: usize,
        height: usize,
        dst_width: usize,
        dst_height: usize,
        color_space: gpui_engine::YuvColorSpace,
    }

    impl Sws {
        fn new(symbols: SwsSymbols) -> Self {
            Self {
                symbols,
                context: std::ptr::null_mut(),
                width: 0,
                height: 0,
                dst_width: 0,
                dst_height: 0,
                color_space: gpui_engine::YuvColorSpace::default(),
            }
        }

        /// The context converting a `width`x`height` `NV12` frame to `dst_width`x`dst_height` `BGRA`
        /// in `color_space`, built if this is the first call or any of them changed. `None` where
        /// libswscale refuses the geometry, so the caller falls back to the scalar walk.
        unsafe fn context_for(
            &mut self,
            width: usize,
            height: usize,
            dst_width: usize,
            dst_height: usize,
            color_space: gpui_engine::YuvColorSpace,
        ) -> Option<*mut c_void> {
            if !self.context.is_null()
                && self.width == width
                && self.height == height
                && self.dst_width == dst_width
                && self.dst_height == dst_height
                && self.color_space == color_space
            {
                return Some(self.context);
            }
            if !self.context.is_null() {
                (self.symbols.free_context)(self.context);
                self.context = std::ptr::null_mut();
            }
            let context = (self.symbols.get_context)(
                width as c_int,
                height as c_int,
                AV_PIX_FMT_NV12,
                dst_width as c_int,
                dst_height as c_int,
                AV_PIX_FMT_BGRA,
                SWS_BILINEAR,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            );
            if context.is_null() {
                return None;
            }
            // The stream's own matrix and range, so the convert matches a scalar walk with the same
            // coefficients; the packed BGRA destination is full range.
            let (matrix, src_range) = sws_colorspace(color_space);
            let in_table = (self.symbols.get_coefficients)(matrix);
            let out_table = (self.symbols.get_coefficients)(matrix);
            (self.symbols.set_colorspace_details)(
                context,
                in_table,
                src_range,
                out_table,
                1,
                0,
                1 << 16,
                1 << 16,
            );
            self.context = context;
            self.width = width;
            self.height = height;
            self.dst_width = dst_width;
            self.dst_height = dst_height;
            self.color_space = color_space;
            Some(context)
        }

        /// Convert a `NV12` frame's planes into packed `BGRA8` at `out`, scaled to
        /// `dst_width`x`dst_height`, or `false` where libswscale will not take the frame and the
        /// caller must fall back.
        ///
        /// SAFETY: the caller guarantees the planes cover `width`x`height` for the strides given, and
        /// that `out` is at least `dst_width * dst_height * 4` bytes.
        #[allow(clippy::too_many_arguments)]
        unsafe fn convert(
            &mut self,
            luma: *const u8,
            luma_stride: c_int,
            chroma: *const u8,
            chroma_stride: c_int,
            width: usize,
            height: usize,
            dst_width: usize,
            dst_height: usize,
            color_space: gpui_engine::YuvColorSpace,
            out: *mut u8,
        ) -> bool {
            let Some(context) =
                self.context_for(width, height, dst_width, dst_height, color_space)
            else {
                return false;
            };
            let src_data = [luma, chroma, std::ptr::null(), std::ptr::null()];
            let src_stride = [luma_stride, chroma_stride, 0, 0];
            let mut dst_data = [out, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()];
            let dst_stride = [(dst_width * 4) as c_int, 0, 0, 0];
            (self.symbols.scale)(
                context,
                src_data.as_ptr(),
                src_stride.as_ptr(),
                0,
                height as c_int,
                dst_data.as_mut_ptr(),
                dst_stride.as_ptr(),
            ) >= 0
        }
    }

    impl Drop for Sws {
        fn drop(&mut self) {
            if !self.context.is_null() {
                // SAFETY: the context is one `sws_getContext` returned.
                unsafe { (self.symbols.free_context)(self.context) };
                self.context = std::ptr::null_mut();
            }
        }
    }

    /// The `SWS_CS_*` id and source-range flag for the engine's colour space, so `sws_scale` uses the
    /// same matrix and range the scalar converter would.
    fn sws_colorspace(color_space: gpui_engine::YuvColorSpace) -> (c_int, c_int) {
        use gpui_engine::{YuvMatrix, YuvRange};

        let matrix = match color_space.matrix {
            YuvMatrix::Bt601 => SWS_CS_ITU601,
            YuvMatrix::Bt709 => SWS_CS_ITU709,
            YuvMatrix::Bt2020 => SWS_CS_BT2020,
        };
        let range = match color_space.range {
            YuvRange::Limited => 0,
            YuvRange::Full => 1,
        };
        (matrix, range)
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

        /// Drop everything the decoder has buffered and queued, so it resumes from a new position
        /// rather than emitting frames from the old one.
        pub fn flush(&mut self) {
            self.queue.clear();
            // SAFETY: `context` is this decoder's open context.
            unsafe { (self.symbols.flush_buffers)(self.context) };
        }

        /// The stream is starting over: flush the decoder, and reset the parser so the bytes fed again
        /// are framed as a fresh stream rather than the tail of the old one. Reused across loops
        /// instead of opening a new decoder, which would build a fresh VA-API device and surface pool
        /// the driver does not hand back.
        pub fn rewind(&mut self) {
            self.flush();
            if !self.parser.is_null() {
                // SAFETY: the parser is this decoder's; closing and re-initing resets its state.
                unsafe {
                    (self.symbols.parser_close)(self.parser);
                    self.parser = (self.symbols.parser_init)(AV_CODEC_ID_H264);
                }
            }
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
            let (pts, colorspace, color_range, height, format) = unsafe {
                (
                    (*frame).pts,
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
                    pts,
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
        /// The frame's presentation timestamp, as the decoder read it from the bitstream.
        pts: i64,
        handle: DmaBufHandle,
    }

    impl Frame {
        /// The dma-buf the renderer imports.
        pub fn handle(&self) -> &DmaBufHandle {
            &self.handle
        }

        /// When this frame is shown, in the stream's own time base, or `None` where the stream names no
        /// timestamp. A player compares this against a seek target to decode forward to it; a raw
        /// elementary stream, which the parser frames without timestamps, carries none.
        pub fn pts(&self) -> Option<i64> {
            (self.pts != AV_NOPTS_VALUE).then_some(self.pts)
        }

        /// Read this frame back to the CPU as packed `BGRA8` — what a player that cannot import the
        /// surface as a texture does instead: copy the decoded pixels out of the driver's buffer and
        /// convert them in software.
        ///
        /// The hardware frame is transferred into a software `NV12` frame first, a GPU→CPU copy that is
        /// exactly the cost zero-copy avoids, and the conversion then walks both planes with the
        /// stream's own matrix and range. The result is `width * height * 4` bytes.
        pub fn to_bgra(&self) -> Option<CpuFrame> {
            self.to_bgra_timed().map(|(frame, _, _)| frame)
        }

        /// [`to_bgra`](Self::to_bgra), reporting how long the GPU→CPU transfer and the software
        /// conversion each took, so a caller can see which dominates.
        pub fn to_bgra_timed(&self) -> Option<(CpuFrame, Duration, Duration)> {
            let symbols = self.symbols;
            // SAFETY: the software frame is allocated here and freed before returning; the hardware
            // frame is this frame's own and alive across the call.
            unsafe {
                let mut software = (symbols.frame_alloc)();
                if software.is_null() {
                    return None;
                }
                (*software).format = AV_PIX_FMT_NV12;
                let transfer_start = Instant::now();
                let transferred = (symbols.hwframe_transfer_data)(software, self.frame, 0) >= 0;
                let transfer = transfer_start.elapsed();
                if !transferred {
                    (symbols.frame_free)(&mut software);
                    return None;
                }
                let width = (*software).width.max(0) as usize;
                let height = (*software).height.max(0) as usize;
                let luma = (*software).data[0];
                let chroma = (*software).data[1];
                let luma_stride = (*software).linesize[0];
                let chroma_stride = (*software).linesize[1];
                let color_space = self.handle.color_space;
                let mut bgra = vec![0u8; width * height * 4];
                // A context of this call's own: `to_bgra` builds a frame per call, so there is no
                // scratch to cache one in; the readback that reuses frames is `Frame::readback`.
                let mut sws = SwsSymbols::resolve(&self._libs).map(Sws::new);
                let convert_start = Instant::now();
                if width > 0 && height > 0 && !luma.is_null() && !chroma.is_null() {
                    convert_nv12_to_bgra(
                        luma,
                        luma_stride,
                        chroma,
                        chroma_stride,
                        width,
                        height,
                        width,
                        height,
                        color_space,
                        &mut bgra,
                        sws.as_mut(),
                    );
                }
                let convert = convert_start.elapsed();
                (symbols.frame_free)(&mut software);
                Some((
                    CpuFrame {
                        width: width as u32,
                        height: height as u32,
                        bgra,
                    },
                    transfer,
                    convert,
                ))
            }
        }
    }

    /// A decoded frame read back to the CPU, packed as `BGRA8` — the shape a player has when it presents
    /// through an uploaded texture rather than a dma-buf the renderer imports.
    pub struct CpuFrame {
        pub width: u32,
        pub height: u32,
        /// `width * height * 4` bytes, `BGRA` per pixel.
        pub bgra: Vec<u8>,
    }

    /// Convert a software `NV12` frame to packed `BGRA8`, using the stream's own matrix and range.
    ///
    /// `NV12` is one luma plane at full size and one interleaved chroma plane at half size in both
    /// dimensions; the two chroma samples a pixel needs are shared by a 2x2 block, so the plane is
    /// walked at half the row rate. Scalar, and the fallback for a machine without libswscale, which
    /// [`Sws`] accelerates through and which the parity test below holds this to.
    fn nv12_to_bgra(
        luma: *const u8,
        luma_stride: c_int,
        chroma: *const u8,
        chroma_stride: c_int,
        width: usize,
        height: usize,
        color_space: gpui_engine::YuvColorSpace,
        out: &mut [u8],
    ) {
        use gpui_engine::{YuvMatrix, YuvRange};

        // The luma/chroma weights; green follows from the other two.
        let (kr, kb) = match color_space.matrix {
            YuvMatrix::Bt601 => (0.2990f32, 0.1140f32),
            YuvMatrix::Bt709 => (0.2126, 0.0722),
            YuvMatrix::Bt2020 => (0.2627, 0.0593),
        };
        let kg = 1.0 - kr - kb;
        let (y_scale, y_offset, c_scale) = match color_space.range {
            YuvRange::Limited => (255.0 / 219.0, 16.0, 255.0 / 224.0),
            YuvRange::Full => (1.0, 0.0, 1.0),
        };
        // The matrix and range folded into 16.16 fixed-point coefficients once, so the inner loop is
        // integer arithmetic — several times cheaper than a float per channel per pixel.
        let fix = |v: f32| (v * 65536.0).round() as i32;
        let yc = fix(y_scale);
        let yo = fix(y_offset * y_scale);
        let r_v = fix(2.0 * (1.0 - kr) * c_scale);
        let b_u = fix(2.0 * (1.0 - kb) * c_scale);
        let g_u = fix(2.0 * kb * (1.0 - kb) / kg * c_scale);
        let g_v = fix(2.0 * kr * (1.0 - kr) / kg * c_scale);

        for row in 0..height {
            // SAFETY: the caller guarantees the planes cover the frame for the strides given; a luma
            // row is `width` bytes and a chroma row is `width` bytes of interleaved U/V.
            let (luma_row, chroma_row) = unsafe {
                (
                    std::slice::from_raw_parts(luma.add(row * luma_stride as usize), width),
                    std::slice::from_raw_parts(chroma.add((row / 2) * chroma_stride as usize), width),
                )
            };
            let out_row = &mut out[row * width * 4..(row + 1) * width * 4];
            for (col, pixel) in out_row.chunks_exact_mut(4).enumerate() {
                let y = (i32::from(luma_row[col]) * yc - yo) >> 16;
                let u = i32::from(chroma_row[col & !1]) - 128;
                let v = i32::from(chroma_row[(col & !1) + 1]) - 128;
                pixel[0] = clamp8(y + ((b_u * u) >> 16));
                pixel[1] = clamp8(y - ((g_u * u + g_v * v) >> 16));
                pixel[2] = clamp8(y + ((r_v * v) >> 16));
                pixel[3] = 255;
            }
        }
    }

    /// Clamp a channel to a byte.
    fn clamp8(value: i32) -> u8 {
        value.clamp(0, 255) as u8
    }

    /// Convert an `NV12` frame's planes into packed `BGRA8` in `out`, through `sws_scale` where a
    /// libswscale context is given and libswscale accepts the frame, and the scalar walk otherwise.
    ///
    /// `sws_scale` scales as it converts, so a `dst` smaller than the source downscales in one pass.
    /// The scalar walk cannot scale, so it is reached only when the destination is the source size.
    ///
    /// SAFETY: the caller guarantees the planes cover `width`x`height` for the strides given, and
    /// that `out` is at least `dst_width * dst_height * 4` bytes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn convert_nv12_to_bgra(
        luma: *const u8,
        luma_stride: c_int,
        chroma: *const u8,
        chroma_stride: c_int,
        width: usize,
        height: usize,
        dst_width: usize,
        dst_height: usize,
        color_space: gpui_engine::YuvColorSpace,
        out: &mut [u8],
        sws: Option<&mut Sws>,
    ) {
        if let Some(sws) = sws {
            if sws.convert(
                luma,
                luma_stride,
                chroma,
                chroma_stride,
                width,
                height,
                dst_width,
                dst_height,
                color_space,
                out.as_mut_ptr(),
            ) {
                return;
            }
        }
        if dst_width == width && dst_height == height {
            nv12_to_bgra(
                luma,
                luma_stride,
                chroma,
                chroma_stride,
                width,
                height,
                color_space,
                out,
            );
        }
    }

    impl Drop for Frame {
        fn drop(&mut self) {
            // SAFETY: the frame is one this decoder allocated; the free returns its surface to the pool.
            unsafe { (self.symbols.frame_free)(&mut self.frame) };
        }
    }

    /// A reusable readback: the software frame a hardware frame is transferred into, and the packed
    /// output buffer, kept across frames so a readback reuses memory instead of allocating and
    /// first-touching fresh pages every call.
    pub struct Readback {
        symbols: Symbols,
        software: *mut AvFrame,
        bgra: Vec<u8>,
        /// The libswscale convert, `None` where it is not installed, in which case the scalar walk
        /// stands in. Kept across reads so its context is built once per stream geometry.
        sws: Option<Sws>,
    }

    /// One readback's result, borrowing the scratch's buffer until the next [`Readback::read`].
    pub struct ReadbackFrame<'a> {
        pub width: u32,
        pub height: u32,
        /// `width * height * 4` bytes, `BGRA` per pixel.
        pub bgra: &'a [u8],
        /// How long the GPU→CPU transfer took.
        pub transfer: Duration,
        /// How long the software conversion took.
        pub convert: Duration,
    }

    impl Frame {
        /// A readback scratch that reuses its buffers, for reading frames of this stream back to the
        /// CPU repeatedly.
        pub fn readback(&self) -> Option<Readback> {
            let symbols = self.symbols;
            let sws = SwsSymbols::resolve(&self._libs).map(Sws::new);
            // SAFETY: allocates a software frame, freed in `Readback::drop`.
            let software = unsafe { (symbols.frame_alloc)() };
            if software.is_null() {
                return None;
            }
            // The transfer fills this frame in the format it names, reused on every later read.
            unsafe { (*software).format = AV_PIX_FMT_NV12 };
            Some(Readback {
                symbols,
                software,
                bgra: Vec::new(),
                sws,
            })
        }
    }

    impl Readback {
        /// Transfer `frame` to the CPU and convert it into this scratch's buffers, reusing them, at
        /// the frame's own size.
        pub fn read(&mut self, frame: &Frame) -> Option<ReadbackFrame<'_>> {
            self.read_to(frame, 0, 0)
        }

        /// [`read`](Self::read), but converting to `dst_width`x`dst_height` — downscaling while it
        /// converts, so a caller that knows the size it will draw at uploads only the pixels it
        /// shows. A size of zero, or a machine without libswscale (whose scalar fallback cannot
        /// scale), keeps the frame's own size; the returned width and height are what was produced.
        pub fn read_to(
            &mut self,
            frame: &Frame,
            dst_width: u32,
            dst_height: u32,
        ) -> Option<ReadbackFrame<'_>> {
            let symbols = self.symbols;
            // SAFETY: the software frame is this scratch's and alive; `frame`'s hardware frame is
            // alive for the borrow.
            unsafe {
                let transfer_start = Instant::now();
                let transferred = (symbols.hwframe_transfer_data)(self.software, frame.frame, 0) >= 0;
                let transfer = transfer_start.elapsed();
                if !transferred {
                    return None;
                }
                let width = (*self.software).width.max(0) as usize;
                let height = (*self.software).height.max(0) as usize;
                let luma = (*self.software).data[0];
                let chroma = (*self.software).data[1];
                let luma_stride = (*self.software).linesize[0];
                let chroma_stride = (*self.software).linesize[1];
                // Scaling is libswscale's alone; without it, the scalar walk converts at the
                // source size, so the output is the source size too.
                let (out_width, out_height) = if self.sws.is_some() && dst_width > 0 && dst_height > 0
                {
                    (dst_width as usize, dst_height as usize)
                } else {
                    (width, height)
                };
                let len = out_width * out_height * 4;
                if self.bgra.len() != len {
                    // Sizes it on the first read and whenever the destination changes; later reads
                    // reuse the same pages, so the conversion writes memory that is already resident.
                    self.bgra.resize(len, 0);
                }
                let convert_start = Instant::now();
                if width > 0 && height > 0 && !luma.is_null() && !chroma.is_null() {
                    convert_nv12_to_bgra(
                        luma,
                        luma_stride,
                        chroma,
                        chroma_stride,
                        width,
                        height,
                        out_width,
                        out_height,
                        frame.handle.color_space,
                        &mut self.bgra,
                        self.sws.as_mut(),
                    );
                }
                let convert = convert_start.elapsed();
                Some(ReadbackFrame {
                    width: out_width as u32,
                    height: out_height as u32,
                    bgra: &self.bgra,
                    transfer,
                    convert,
                })
            }
        }
    }

    impl Drop for Readback {
        fn drop(&mut self) {
            // SAFETY: the software frame is one this scratch allocated.
            unsafe { (self.symbols.frame_free)(&mut self.software) };
        }
    }

    /// A staged readback's result: the transfer, the copy into the staging buffer, and the conversion
    /// from it, each timed.
    pub struct StagedReadback<'a> {
        pub width: u32,
        pub height: u32,
        pub bgra: &'a [u8],
        pub transfer: Duration,
        pub copy: Duration,
        pub convert: Duration,
    }

    impl Readback {
        /// Like [`read`](Self::read), but first copying the transferred planes into `staging` (tightly
        /// packed, so a caller can pool it) and converting from there — a test of whether the
        /// transferred buffer itself is slow to read. The source is read once for the copy; the
        /// conversion then reads cached memory.
        pub fn read_staged(
            &mut self,
            frame: &Frame,
            staging: &mut Vec<u8>,
        ) -> Option<StagedReadback<'_>> {
            let symbols = self.symbols;
            // SAFETY: the software frame is this scratch's and alive; `frame`'s hardware frame is
            // alive for the borrow; the staging offsets are within the resized buffer.
            unsafe {
                let transfer_start = Instant::now();
                let transferred = (symbols.hwframe_transfer_data)(self.software, frame.frame, 0) >= 0;
                let transfer = transfer_start.elapsed();
                if !transferred {
                    return None;
                }
                let width = (*self.software).width.max(0) as usize;
                let height = (*self.software).height.max(0) as usize;
                let luma = (*self.software).data[0];
                let chroma = (*self.software).data[1];
                let luma_stride = (*self.software).linesize[0] as usize;
                let chroma_stride = (*self.software).linesize[1] as usize;
                let luma_len = width * height;
                let chroma_len = luma_len / 2;

                let copy_start = Instant::now();
                staging.clear();
                staging.resize(luma_len + chroma_len, 0);
                if !luma.is_null() && width > 0 {
                    for row in 0..height {
                        let src = std::slice::from_raw_parts(luma.add(row * luma_stride), width);
                        staging[row * width..row * width + width].copy_from_slice(src);
                    }
                }
                if !chroma.is_null() && width > 0 {
                    for row in 0..height / 2 {
                        let src = std::slice::from_raw_parts(chroma.add(row * chroma_stride), width);
                        let at = luma_len + row * width;
                        staging[at..at + width].copy_from_slice(src);
                    }
                }
                let copy = copy_start.elapsed();

                let len = width * height * 4;
                if self.bgra.len() != len {
                    self.bgra.resize(len, 0);
                }
                let convert_start = Instant::now();
                if width > 0 && height > 0 {
                    convert_nv12_to_bgra(
                        staging.as_ptr(),
                        width as c_int,
                        staging.as_ptr().add(luma_len),
                        width as c_int,
                        width,
                        height,
                        width,
                        height,
                        frame.handle.color_space,
                        &mut self.bgra,
                        self.sws.as_mut(),
                    );
                }
                let convert = convert_start.elapsed();
                Some(StagedReadback {
                    width: width as u32,
                    height: height as u32,
                    bgra: &self.bgra,
                    transfer,
                    copy,
                    convert,
                })
            }
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
        /// The video stream's index in the container, so a seek targets that stream's time base.
        index: c_int,
        /// The stream's average frame rate, for pacing playback.
        frame_rate: AvRational,
        /// The stream's time base, so a seek target can be expressed in the units a frame's `pts` is in.
        time_base: AvRational,
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
                index: -1,
                frame_rate: AvRational { num: 0, den: 0 },
                time_base: AvRational { num: 0, den: 0 },
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
            demuxer.index = index;
            demuxer.frame_rate = stream.avg_frame_rate;
            demuxer.time_base = stream.time_base;
            demuxer.packet = (demuxer.symbols.packet_alloc)();
            anyhow::ensure!(!demuxer.packet.is_null(), "av_packet_alloc failed");
            Ok(demuxer)
        }

        /// Open a decoder over this container's video stream, on the libraries the demuxer loaded.
        pub fn decoder(&self) -> Option<Decoder> {
            Decoder::open_demuxed(self.libs.clone(), self.parameters)
        }

        /// Read the next video packet and hand it to `decoder` — a container frames its packets
        /// already, so they go in ahead of any parser. Packets from the container's other streams
        /// (audio, subtitles) are skipped: a container interleaves them, and only the video feeds the
        /// decoder. `Ok(false)` at the end of the stream; an error only where the decoder refused a
        /// packet.
        pub fn send_next(&mut self, decoder: &mut Decoder) -> anyhow::Result<bool> {
            loop {
                // SAFETY: the packet is one this demuxer allocated; releasing it before re-reading
                // keeps the next read from overwriting a buffer still in use, and it accepts a
                // released packet.
                unsafe { (self.symbols.packet_unref)(self.packet) };
                if unsafe { (self.symbols.read_frame)(self.format, self.packet) } < 0 {
                    return Ok(false);
                }
                if unsafe { (*self.packet).stream_index } == self.index {
                    decoder.send_packet(self.packet)?;
                    return Ok(true);
                }
                // Another stream's packet: read on until the video's arrives (or the stream ends).
            }
        }

        /// The stream's average frame period, where the container declares one, so a player can pace
        /// playback at the rate the stream was authored at.
        pub fn frame_period(&self) -> Option<Duration> {
            let (num, den) = (self.frame_rate.num, self.frame_rate.den);
            (num > 0 && den > 0).then(|| Duration::from_secs_f64(f64::from(den) / f64::from(num)))
        }

        /// How long the container is, where it declares a duration, so a player can show progress.
        pub fn duration(&self) -> Option<Duration> {
            // SAFETY: `format` is this demuxer's open context, and `duration` is read at the offset
            // the declared ABI is asserted against.
            let duration = unsafe { (&*self.format).duration };
            (duration > 0).then(|| Duration::from_micros(duration as u64))
        }

        /// Convert a playback position — an offset from the stream's start — into the timestamp the
        /// stream's frames are measured in, so a seek target can be compared against a frame's `pts`.
        ///
        /// This is the inverse of what libavformat does when [`seek`](Self::seek) is given a
        /// timestamp in `AV_TIME_BASE` units with a `-1` stream index, so the target it produces is in
        /// the same units as the `pts` the decoder then reports.
        pub fn timestamp_at(&self, at: Duration) -> i64 {
            let (num, den) = (self.time_base.num as i128, self.time_base.den as i128);
            if num <= 0 || den <= 0 {
                return AV_NOPTS_VALUE;
            }
            // `pts = at * den / (num * 1e6)`; the `i128` keeps a long stream's product from wrapping.
            let micros = at.as_micros() as i128;
            (micros * den / (num * 1_000_000)).clamp(i64::MIN as i128, i64::MAX as i128) as i64
        }

        /// Move the read position to the keyframe at or before `to`, returning the target in the
        /// stream's own time base — the value a later frame's `pts` is compared against to decode
        /// forward to it — or `None` where the source cannot seek. The decoder must be flushed before
        /// it is fed again: a seek invalidates everything it has buffered.
        pub fn seek(&mut self, to: Duration) -> Option<i64> {
            let timestamp = self.timestamp_at(to);
            if timestamp == AV_NOPTS_VALUE {
                return None;
            }
            // SAFETY: `format` is this demuxer's open context; the timestamp is in the video stream's
            // own time base, the one its frames report, so both name the same instant.
            let seeked = unsafe {
                (self.symbols.seek_frame)(self.format, self.index, timestamp, AVSEEK_FLAG_BACKWARD)
            };
            (seeked >= 0).then_some(timestamp)
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

        /// The libswscale convert, where installed, agrees with the scalar walk it stands in for —
        /// the same matrix and range, to within a rounding step. A swapped matrix or range would
        /// diverge by tens of levels, so a small tolerance still catches a real divergence.
        #[test]
        fn swscale_and_the_scalar_walk_agree() {
            use std::ffi::c_int;

            use super::super::ffmpeg;
            use super::{nv12_to_bgra, Sws, SwsSymbols};

            let Some(libs) = ffmpeg::load() else {
                eprintln!("skipping: libavcodec/libavutil of the declared ABI is unavailable");
                return;
            };
            let Some(symbols) = SwsSymbols::resolve(&libs) else {
                eprintln!("skipping: libswscale of the declared ABI is unavailable");
                return;
            };

            // A synthetic NV12 frame: a luma ramp, and chroma off neutral so both the U and V
            // coefficients are exercised.
            let (width, height) = (16usize, 8usize);
            let mut nv12 = vec![0u8; width * height + width * height / 2];
            for (i, y) in nv12[..width * height].iter_mut().enumerate() {
                *y = ((i * 7) % 256) as u8;
            }
            let (luma, chroma) = nv12.split_at_mut(width * height);
            for (i, c) in chroma.iter_mut().enumerate() {
                *c = if i % 2 == 0 { 135 } else { 72 };
            }

            for color_space in [YuvColorSpace::BT601_LIMITED, YuvColorSpace::BT709_LIMITED] {
                let mut expected = vec![0u8; width * height * 4];
                nv12_to_bgra(
                    luma.as_ptr(),
                    width as c_int,
                    chroma.as_ptr(),
                    width as c_int,
                    width,
                    height,
                    color_space,
                    &mut expected,
                );

                let mut got = vec![0u8; width * height * 4];
                let mut sws = Sws::new(symbols);
                // SAFETY: the planes cover the frame for the strides given, and `got` is sized.
                let converted = unsafe {
                    sws.convert(
                        luma.as_ptr(),
                        width as c_int,
                        chroma.as_ptr(),
                        width as c_int,
                        width,
                        height,
                        width,
                        height,
                        color_space,
                        got.as_mut_ptr(),
                    )
                };
                assert!(converted, "libswscale refused the frame");

                for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
                    assert!(
                        a.abs_diff(*b) <= 2,
                        "channel {i} differs by more than a rounding step: {a} vs {b}"
                    );
                }
            }
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

    /// A seek lands on the frame *at* the target, not the keyframe before it. A container seeks to
    /// the keyframe at or before the target, so the frames between are decoded forward and dropped
    /// until the target is reached — which is what makes a scrub land where it was dragged to on a
    /// stream whose keyframes are sparse. The clip's keyframes fall on even frames, so an odd frame is
    /// one the decoder can only reach by decoding forward from the keyframe before it.
    #[test]
    fn a_seek_decodes_forward_to_the_frame_at_the_target() {
        let Some(probe) = Demuxer::open(Path::new(FIXTURE_MP4)) else {
            eprintln!("skipping: libavformat of the declared ABI is unavailable");
            return;
        };
        let Some(mut playback) = Playback::open_path(Path::new(FIXTURE_MP4), 3) else {
            eprintln!(
                "skipping: libavformat/libavcodec of the declared ABI, or the VA-API device, is \
                 unavailable"
            );
            return;
        };
        let period = playback.frame_period().expect("the clip declares a frame rate");
        let period_ts = probe.timestamp_at(period);

        for frame in [1u32, 7, 13, 21, 31, 45] {
            let target = period * frame;
            let target_ts = probe.timestamp_at(target);
            assert!(playback.seek(target), "seek to {target:?}");
            let handle = playback.next().expect("a frame after the seek");
            let pts = playback
                .live
                .back()
                .and_then(|live| live._frame.pts())
                .expect("the frame shown carries a timestamp");
            // Release the frame, as the renderer does, so `live` never fills to `depth`.
            signal(&handle);
            assert!(
                pts >= target_ts && pts < target_ts + period_ts,
                "seeking to frame {frame} shows it, not the keyframe before: pts {pts}, target {target_ts}"
            );
        }
    }
}
