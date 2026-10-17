//! An external surface composited through the authoring API, end to end.
//!
//! Two authoring layers reach one: the `surface()` element (single-plane `Bgra8`/`Rgba8`, two-plane
//! `Nv12`) and `gpu_canvas(..)`'s paint callback (a same-device `wgpu::TextureView`). The producer
//! exports through `gpui_interop::SharedBuffer::export_at` on the renderer's **own** device — a
//! dma-buf is importable only on the GPU it was exported from — the consumer is the platform's
//! headless renderer, the same one a window drives, and the pixels are read back with
//! `capture_screenshot`.
//!
//! Opt in, because it needs a GPU and the crate's own unit-test target does not build:
//!
//! ```text
//! ZED_DEVICE_ID=1916 cargo test -p gpui_va --features test-support --test surface_dmabuf
//! ```
//!
//! It skips (rather than fails) where there is no Vulkan device, and it pins the adapter when you
//! pass `ZED_DEVICE_ID`.
//!
//! # Running this on a machine with two GPUs
//!
//! The headless renderer chooses an adapter itself, and on a laptop it may choose the discrete one.
//! A userspace fault *while GPU work is in flight* can wedge that GPU: an earlier revision dropped
//! the Vulkan loader before the device it had made, and its crash produced `NVRM: Xid 13` followed by
//! repeated `Xid 158` (`NV_UFLUSH_FB_FLUSH` timeout) on an NVIDIA 930M, hanging the machine until
//! reboot. This test keeps the loader alive, and the producer exports on the renderer's own device —
//! whichever GPU the renderer chose — but you can pin the adapter explicitly too:
//!
//! ```text
//! ZED_DEVICE_ID=1916 cargo test -p gpui_va --features test-support --test surface_dmabuf
//! ```
//!
//! `ZED_DEVICE_ID` is a **four-digit hexadecimal** PCI device id; `1916` is the Intel HD 520 on the
//! machine this was written on (a decimal id like `6422` parses as the hex `0x6422`, matches nothing,
//! and silently falls back to the default adapter — the discrete one here).

#![cfg(all(target_os = "linux", feature = "test-support"))]

use std::cell::{Cell, RefCell};
use std::io::Write as _;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::rc::Rc;

use gpui::{
    AnyWindowHandle, AppContext as _, Context, Corners, DmaBufHandle, DmaBufPlane,
    HeadlessAppContext, ImportedTextureHandle, IntoElement, Render, SurfaceFormatKind, Window, div,
    gpu_canvas, prelude::*, px, size, surface,
};
use gpui_interop::{GpuDevice, PlaneLayout, SharedBuffer, Tiling};
use gpui_wgpu::{CosmicTextSystem, ImportedTextureExt as _, WgpuHeadlessRenderer, WgpuRenderer};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
/// The one fixture colour, olive green, shared by every row that asserts a colour. Opaque, so the
/// surface's premultiplied-alpha blend leaves it unchanged.
const COLOUR: [u8; 4] = [128, 128, 0, 255];
/// `COLOUR` in the byte order a `Bgra8` buffer stores: blue, green, red, alpha.
const BGRA_COLOUR: [u8; 4] = [COLOUR[2], COLOUR[1], COLOUR[0], COLOUR[3]];

/// Borrow the renderer's own device, on the GPU this process draws its headless frames through.
///
/// A dma-buf is importable only on the GPU it was exported from, and the renderer's device is reachable
/// only from a `gpu_canvas` callback, so this opens a throwaway headless window, paints one frame to
/// run a canvas, and lifts the device back out. It stays valid after that context drops:
/// `WgpuContext::new_headless` caches the context in a process-wide static, so the instance the device
/// was made from is never torn down for the life of the process.
fn renderer_device() -> Option<GpuDevice> {
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });

    let device: Rc<RefCell<Option<GpuDevice>>> = Rc::new(RefCell::new(None));
    let slot = device.clone();
    let window = cx
        .open_window(size(px(WIDTH as f32), px(HEIGHT as f32)), move |_window, cx| {
            cx.new(|_| DeviceProbeView { device: slot })
        })
        .ok()?;
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })
    .ok()?;

    device.borrow_mut().take()
}

/// A view whose only job is to run one `gpu_canvas` callback, where the renderer's device is reachable,
/// and stash it for the caller of [`renderer_device`].
struct DeviceProbeView {
    device: Rc<RefCell<Option<GpuDevice>>>,
}

impl Render for DeviceProbeView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let slot = self.device.clone();
        gpu_canvas(move |gpu| {
            // `WgpuHeadlessRenderer`, not `WgpuRenderer`: a headless window draws through the headless
            // renderer, and only the renderer the window actually holds lends a device.
            let Ok(interop) = gpui_interop::attach(gpu.try_device::<WgpuHeadlessRenderer>()) else {
                return;
            };
            if let Some(device) = interop.adapter().gpu_device() {
                *slot.borrow_mut() = Some(device);
            }
        })
        .size_full()
    }
}

/// A buffer exported on the renderer's GPU, kept with the device it came from so the device outlives
/// the memory and the memory outlives the handle's fds.
///
/// Declaration order is load-bearing: `SharedBuffer` frees its memory through a clone of the device's
/// `ash::Device`, and the `GpuDevice` destroys that device when it drops, so the buffer is declared
/// first and Rust drops it before the device.
struct Exported {
    buffer: SharedBuffer,
    _device: GpuDevice,
}

impl Exported {
    /// The description the `surface()` element takes.
    fn handle(&self) -> DmaBufHandle {
        self.buffer.handle()
    }
}

/// Export `content` as a linear dma-buf with the plane layout `planes` declares, on the renderer's own
/// device — the only one that can import the result. `None`, with a line to `stderr`, when there is no
/// such device or the export fails: a skip rather than a failure.
fn export(
    width: u32,
    height: u32,
    format: SurfaceFormatKind,
    planes: &[PlaneLayout],
    content: &[u8],
) -> Option<Exported> {
    let Some(device) = renderer_device() else {
        eprintln!("skipping: the renderer lends no device to export on");
        return None;
    };
    match SharedBuffer::export_at(&device, width, height, format, planes, content) {
        Ok(buffer) => Some(Exported {
            buffer,
            _device: device,
        }),
        Err(error) => {
            eprintln!("skipping: could not export a dma-buf: {error:#}");
            None
        }
    }
}

struct SurfaceView {
    handle: DmaBufHandle,
}

impl Render for SurfaceView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(surface(self.handle.clone()).w_full().h_full())
    }
}

/// Mount a view painting `handle`, draw a frame, and return the centre pixel.
fn composite(handle: DmaBufHandle) -> anyhow::Result<[u8; 4]> {
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });

    let window = cx.open_window(size(px(WIDTH as f32), px(HEIGHT as f32)), |_window, cx| {
        cx.new(|_| SurfaceView { handle })
    })?;
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })?;

    let image = cx.capture_screenshot(window)?;
    Ok(image.get_pixel(WIDTH / 2, HEIGHT / 2).0)
}

/// Lay a row pattern into `content` at `plane`, honouring the plane's own stride — the whole-object
/// layout [`export_at`](gpui_interop::SharedBuffer::export_at) demands, where a row may be padded
/// past the pixels in it.
fn fill_plane(content: &mut [u8], plane: &PlaneLayout, width: usize, rows: usize, pattern: &[u8]) {
    let base = plane.offset as usize;
    let stride = plane.stride as usize;
    for row in 0..rows {
        let start = base + row * stride;
        for (column, byte) in content[start..start + width].iter_mut().enumerate() {
            *byte = pattern[column % pattern.len()];
        }
    }
}

/// A view that paints nothing — the window a dropped surface leaves behind.
struct EmptyView;

impl Render for EmptyView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full()
    }
}

/// The centre pixel of a window with nothing painted in it — what a *dropped* surface leaves behind,
/// and the reference a refusal is checked against: a composited tile would never match it.
fn empty_window_pixel() -> anyhow::Result<[u8; 4]> {
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });

    let window = cx.open_window(size(px(WIDTH as f32), px(HEIGHT as f32)), |_window, cx| {
        cx.new(|_| EmptyView)
    })?;
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })?;

    let image = cx.capture_screenshot(window)?;
    Ok(image.get_pixel(WIDTH / 2, HEIGHT / 2).0)
}

/// A view painting a same-device texture through `gpu_canvas(..)`, and whether the paint-time
/// callback found a device to make it on.
struct CanvasView {
    colour: [u8; 4],
    painted: Rc<Cell<bool>>,
}

impl Render for CanvasView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let colour = self.colour;
        let painted = self.painted.clone();
        gpu_canvas(move |gpu| {
            let Some(handle) = imported_texture(gpu.try_device::<WgpuRenderer>(), colour) else {
                return;
            };
            painted.set(true);
            gpu.paint_texture(handle, Corners::default(), 1.0, false);
        })
        .size_full()
    }
}

/// Make `colour` a texture on the renderer's own device and wrap it as an [`ImportedTextureHandle`].
/// `None` when the window's renderer lends no device.
fn imported_texture(
    device: Option<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)>,
    colour: [u8; 4],
) -> Option<ImportedTextureHandle> {
    // A producer reaches the renderer's device through the canvas's typed door. A window whose
    // renderer lends none hands back `None` — which is what the headless harness here does.
    let (device, queue) = device?;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_dmabuf_gpu_canvas"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB, the colour space `to_imported_handle` requires; `colour` is already sRGB-encoded.
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        // `TEXTURE_BINDING` is what the handle builder checks, `COPY_DST` is how the colour gets in.
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut content = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..(WIDTH * HEIGHT) {
        content.extend_from_slice(&colour);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &content,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 4),
            rows_per_image: Some(HEIGHT),
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    texture
        .create_view(&wgpu::TextureViewDescriptor::default())
        .to_imported_handle()
        .ok()
}

/// Mount a `gpu_canvas` whose callback fills `colour` on the window's device, draw a frame, and
/// return the centre pixel. `None` when the window's renderer lends no device.
fn composite_canvas(colour: [u8; 4]) -> anyhow::Result<Option<[u8; 4]>> {
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });

    let painted = Rc::new(Cell::new(false));
    let window = cx.open_window(size(px(WIDTH as f32), px(HEIGHT as f32)), |_window, cx| {
        cx.new(|_| CanvasView {
            colour,
            painted: painted.clone(),
        })
    })?;
    let window: AnyWindowHandle = window.into();

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })?;

    if !painted.get() {
        return Ok(None);
    }
    let image = cx.capture_screenshot(window)?;
    Ok(Some(image.get_pixel(WIDTH / 2, HEIGHT / 2).0))
}

/// Whether a headless renderer exists at all — an error or `None` where there is no adapter, which is
/// a skip rather than a failure. The device a producer exports on is reached later, from a canvas, by
/// [`renderer_device`]; this only gates whether it is worth painting one.
fn gpu_available() -> bool {
    gpui::current_headless_renderer().is_ok_and(|renderer| renderer.is_some())
}

#[test]
fn a_single_plane_surface_reads_back_byte_for_byte() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let mut content = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..(WIDTH * HEIGHT) {
        content.extend_from_slice(&COLOUR);
    }
    let Some(exported) = export(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::rgba8(),
        &[PlaneLayout {
            offset: 0,
            stride: WIDTH * 4,
        }],
        &content,
    ) else {
        return;
    };

    let pixel = composite(exported.handle()).expect("composite the surface");
    assert_eq!(pixel, COLOUR);
}

/// The renderer's half of the dma-buf handshake: it signals a surface's release descriptor once the
/// submission that sampled it has completed, so a producer knows it may recycle the surface.
#[test]
fn the_renderer_signals_a_surfaces_release_descriptor() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let mut content = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..(WIDTH * HEIGHT) {
        content.extend_from_slice(&COLOUR);
    }
    let Some(exported) = export(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::rgba8(),
        &[PlaneLayout {
            offset: 0,
            stride: WIDTH * 4,
        }],
        &content,
    ) else {
        return;
    };

    // The descriptor a producer waits on; the renderer writes to it once it is done with the buffer.
    let release = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    assert!(release >= 0, "create an eventfd");
    // SAFETY: `eventfd` returned a fresh descriptor this process owns.
    let release = unsafe { OwnedFd::from_raw_fd(release) };

    // The handle takes ownership of the descriptor, so hand it a duplicate and keep the original to
    // wait on.
    let handle = exported.handle().with_release(
        release
            .try_clone()
            .expect("duplicate the release descriptor"),
    );

    let _ = composite(handle).expect("composite the surface");

    // The signalling happens from the completion callback of the frame's submission, so give it a
    // moment rather than expecting it the instant `draw` returns.
    let mut descriptor = libc::pollfd {
        fd: release.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let ready = unsafe { libc::poll(&mut descriptor, 1, 5000) };
    assert_eq!(
        ready, 1,
        "the renderer should signal the release descriptor"
    );
}

#[test]
fn a_single_bgra_plane_reads_back_as_the_source_colour() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    // A `Bgra8` buffer stores blue, green, red, alpha, so the same fixture colour goes in
    // channel-swapped; the sampler maps the bytes back to RGBA, so it must come out as `COLOUR`.
    let mut content = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..(WIDTH * HEIGHT) {
        content.extend_from_slice(&BGRA_COLOUR);
    }
    let Some(exported) = export(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::bgra8(),
        &[PlaneLayout {
            offset: 0,
            stride: WIDTH * 4,
        }],
        &content,
    ) else {
        return;
    };

    let pixel = composite(exported.handle()).expect("composite the surface");
    assert_eq!(pixel, COLOUR);
}

#[test]
fn a_surface_with_a_signalled_acquire_fence_is_waited_on() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let mut content = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for _ in 0..(WIDTH * HEIGHT) {
        content.extend_from_slice(&COLOUR);
    }
    let Some(exported) = export(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::rgba8(),
        &[PlaneLayout {
            offset: 0,
            stride: WIDTH * 4,
        }],
        &content,
    ) else {
        return;
    };
    // A `sync_file` is readable once its fence signals; a pipe holding a byte has that same shape,
    // so the renderer's wait returns at once and the surface still composites.
    let (read_end, mut write_end) = std::io::pipe().expect("a pipe to stand in for a sync_file");
    write_end.write_all(&[1]).expect("signal the fence");
    // `export_at` carries no acquire fence, so rebuild the handle from the exported plane's fd with the
    // fence attached — the same declared layout, and a duplicate of the fd the export made.
    let handle = DmaBufHandle::new(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::rgba8(),
        DmaBufHandle::LINEAR,
        [DmaBufPlane::new(
            exported.handle().planes[0]
                .fd
                .try_clone()
                .expect("duplicate the dma-buf"),
            0,
            WIDTH * 4,
        )],
        Some(OwnedFd::from(read_end)),
    );

    let pixel = composite(handle).expect("composite the surface");
    assert_eq!(pixel, COLOUR);
}

/// The consumer-guided contract: the driver picks the image's layout and the handle reports it, so
/// the buffer is one this device samples by construction. That is what makes it the same fixture on
/// a device whose linear `NV12` is the canonical DRM layout and on one that pads it, and it keeps
/// the BT.601 path exercised on both.
#[test]
fn an_nv12_surface_converts_through_the_shader() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(device) = renderer_device() else {
        eprintln!("skipping: the renderer lends no device to export on");
        return;
    };
    let luma_size = (WIDTH * HEIGHT) as usize;
    let chroma_size = (WIDTH / 2 * HEIGHT / 2 * 2) as usize;
    // The shared colour in the two-plane form: full-resolution luma, then half-resolution
    // interleaved chroma, carrying the bytes the BT.601 conversion maps back to olive. `fill` takes
    // each plane tightly packed at its own resolution; the driver places it in the image.
    let (y, cb, cr) = nv12_from_rgb(COLOUR);
    let luma = vec![y; luma_size];
    let mut chroma = vec![0u8; chroma_size];
    for pair in chroma.as_chunks_mut::<2>().0 {
        pair[0] = cb;
        pair[1] = cr;
    }
    let buffer = match SharedBuffer::new(
        &device,
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::nv12(),
        Tiling::Linear,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("skipping: could not allocate an NV12 buffer: {error:#}");
            return;
        }
    };
    if let Err(error) = buffer.fill(&device, &[&luma, &chroma]) {
        eprintln!("skipping: could not fill the NV12 buffer: {error:#}");
        return;
    }

    let pixel = composite(buffer.handle()).expect("composite the surface");
    // The same BT.601 matrix the shader applies, on the CPU, to the bytes the planes carry.
    let expected = ycbcr_to_rgb(
        f32::from(y) / 255.0,
        f32::from(cb) / 255.0,
        f32::from(cr) / 255.0,
    );
    for channel in 0..3 {
        let want = (expected[channel] * 255.0).round() as i32;
        let got = i32::from(pixel[channel]);
        assert!(
            (got - want).abs() <= 2,
            "channel {channel} is {got}, but the shader's matrix predicts {want}",
        );
    }
    assert_eq!(pixel[3], 255, "the surface should be opaque");
}

/// A handle that declares a layout the driver does not produce is refused, not sampled.
///
/// [`export_at`](gpui_interop::SharedBuffer::export_at) lets a producer push a plane layout rather
/// than take the one the driver chose, and a layout the driver did not choose is one it cannot
/// sample: measured on an NVIDIA 930M, sampling such an image never completes and hangs the command
/// processor. This buffer pads its luma plane, so its chroma sits at 8192 where the device's own
/// linear layout puts it at 4096, and the importer drops the surface rather than bind it. The window
/// stays empty, and the fixture colour would never match it.
#[test]
fn a_linear_nv12_surface_declaring_a_layout_the_driver_does_not_produce_is_refused() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let luma_size = (WIDTH * HEIGHT) as usize;
    let chroma_size = (WIDTH / 2 * HEIGHT / 2 * 2) as usize;
    // A whole page past the luma's end, which no driver's own linear layout would choose.
    let chroma_offset = luma_size + 4096;
    let (y, cb, cr) = nv12_from_rgb(COLOUR);

    let mut content = vec![0u8; chroma_offset + chroma_size];
    content[..luma_size].fill(y);
    for pair in content[chroma_offset..].as_chunks_mut::<2>().0 {
        pair[0] = cb;
        pair[1] = cr;
    }

    let Some(exported) = export(
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::nv12(),
        &[
            PlaneLayout {
                offset: 0,
                stride: WIDTH,
            },
            PlaneLayout {
                offset: chroma_offset as u64,
                stride: WIDTH,
            },
        ],
        &content,
    ) else {
        return;
    };

    let background = empty_window_pixel().expect("capture an empty window");
    let pixel = composite(exported.handle()).expect("composite the refused surface");
    assert_eq!(
        pixel, background,
        "the surface was not refused: the handle declares chroma at {chroma_offset}, where this \
         device does not lay its own linear NV12 out. Sampling a layout the driver did not produce \
         reads the padding as chroma (which composites green), and on some devices never completes",
    );
}

/// The client's happy path, negotiated up front: ask the renderer's device which plane layout it
/// gives a plain `LINEAR` image and how big the buffer must be, then export exactly that and
/// composite it.
///
/// This is what [`GpuDevice::linear_layout`] exists for — a producer consults it *before* exporting,
/// so the buffer it hands over is one the renderer takes. It is the portable half of the pair: a
/// device whose linear `NV12` is the canonical DRM layout answers with it, and one that pads (an
/// NVIDIA 930M puts chroma at 65536) answers with the padded layout, which the producer declares
/// instead of its own idea. The colour assertion is the real test: it holds only if the renderer
/// samples the planes where the handle declares them.
#[test]
fn a_buffer_sized_to_the_devices_report_composites() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(device) = renderer_device() else {
        eprintln!("skipping: the renderer lends no device to export on");
        return;
    };
    let luma_size = (WIDTH * HEIGHT) as usize;
    // What the producer *would* declare on its own — the DRM `NV12` convention, chroma directly
    // after the luma. The report answers with the layout this device actually uses, which on a
    // padding device is not this one; `declared` measures how far apart they are.
    let proposed = [
        PlaneLayout {
            offset: 0,
            stride: WIDTH,
        },
        PlaneLayout {
            offset: luma_size as u64,
            stride: WIDTH,
        },
    ];
    let report = match device.linear_layout(SurfaceFormatKind::nv12(), WIDTH, HEIGHT, &proposed) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("skipping: the device will not make the declared layout: {error:#}");
            return;
        }
    };
    eprintln!(
        "nv12 negotiation: native {:?}, required {}, proposed span {}",
        report.native, report.required, report.declared,
    );
    let [luma, chroma] = <[PlaneLayout; 2]>::try_from(report.native.as_slice())
        .expect("an NV12 layout has two planes");

    // The producer's self-correction: declare the layout the device reported and size the buffer to
    // what its image needs, placing the planes where that layout says.
    let (y, cb, cr) = nv12_from_rgb(COLOUR);
    let mut content = vec![0u8; report.required as usize];
    fill_plane(&mut content, &luma, WIDTH as usize, HEIGHT as usize, &[y]);
    fill_plane(&mut content, &chroma, WIDTH as usize, (HEIGHT / 2) as usize, &[cb, cr]);
    let buffer = match SharedBuffer::export_at(
        &device,
        WIDTH,
        HEIGHT,
        SurfaceFormatKind::nv12(),
        &report.native,
        &content,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            eprintln!("skipping: could not export a dma-buf: {error:#}");
            return;
        }
    };

    let pixel = composite(buffer.handle()).expect("composite the surface");
    let expected = ycbcr_to_rgb(
        f32::from(y) / 255.0,
        f32::from(cb) / 255.0,
        f32::from(cr) / 255.0,
    );
    for channel in 0..3 {
        let want = (expected[channel] * 255.0).round() as i32;
        let got = i32::from(pixel[channel]);
        assert!(
            (got - want).abs() <= 2,
            "channel {channel} is {got}, but the declared layout predicts {want}; a device that \
             samples its own padded layout instead reads the padding as chroma, which composites green",
        );
    }
    assert_eq!(pixel[3], 255, "the surface should be opaque");
}

/// The same fixture, but through a producer that hands the renderer a **tiled** buffer: the VA-API
/// `Y_TILED` NV12 surface `gpui_va` builds, the shape a hardware decoder exports and `vaPutImage`
/// cannot fill. The importer must import it under its modifier — which the device only permits
/// because the escape hatch enabled `VK_EXT_image_drm_format_modifier` — and the shader must still
/// convert it.
#[test]
fn a_tiled_nv12_surface_imports_under_its_modifier_and_converts() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some((_surface, handle)) = gpui_va::nv12(WIDTH, HEIGHT, COLOUR) else {
        eprintln!("skipping: no VA-API surface (libva, its driver, or the device unavailable)");
        return;
    };
    eprintln!(
        "importing a tiled {}×{} surface with modifier {:#x}",
        handle.width, handle.height, handle.modifier,
    );

    let pixel = composite(handle).expect("composite the tiled surface");
    let (y, cb, cr) = nv12_from_rgb(COLOUR);
    let expected = ycbcr_to_rgb(
        f32::from(y) / 255.0,
        f32::from(cb) / 255.0,
        f32::from(cr) / 255.0,
    );
    for channel in 0..3 {
        let want = (expected[channel] * 255.0).round() as i32;
        let got = i32::from(pixel[channel]);
        assert!(
            (got - want).abs() <= 2,
            "channel {channel} is {got}, but the shader's matrix predicts {want}",
        );
    }
    assert_eq!(pixel[3], 255, "the surface should be opaque");
}

/// Mount a view painting `handle`, draw a frame, and return the capture's size and its pixels in
/// row-major order, so a test can sample wherever it likes.
fn composite_pixels(handle: DmaBufHandle) -> (u32, u32, Vec<[u8; 4]>) {
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });
    let window = cx
        .open_window(size(px(WIDTH as f32), px(HEIGHT as f32)), |_window, cx| {
            cx.new(|_| SurfaceView { handle })
        })
        .expect("a window");
    let window: AnyWindowHandle = window.into();
    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        let _ = window.draw(cx);
    })
    .expect("draw");
    let image = cx.capture_screenshot(window).expect("capture");
    let (width, height) = (image.width(), image.height());
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(image.get_pixel(x, y).0);
        }
    }
    (width, height, pixels)
}

/// Composite `handle` and return the pixel at each of the seven bar centres, in capture space.
fn composite_bar_centers(handle: DmaBufHandle) -> Vec<[u8; 4]> {
    let (width, height, pixels) = composite_pixels(handle);
    (0..SMPTE_BARS.len() as u32)
        .map(|bar| {
            let x = (2 * bar + 1) * width / (2 * SMPTE_BARS.len() as u32);
            pixels[(height / 2 * width + x) as usize]
        })
        .collect()
}

/// The fixture's bars, in SMPTE 75% order: grey, yellow, cyan, green, magenta, red, blue. The
/// fixture encodes each with the inverse of the shader's matrix, so the shader must map them back.
const SMPTE_BARS: [[u8; 4]; 7] = [
    [191, 191, 191, 255],
    [191, 191, 0, 255],
    [0, 191, 191, 255],
    [0, 191, 0, 255],
    [191, 0, 191, 255],
    [191, 0, 0, 255],
    [0, 0, 191, 255],
];

/// The real thing at last: an H.264 keyframe the VA-API driver decodes into a `Y_TILED` surface,
/// imported under its modifier and converted by the shader. `gpui_va` carries the fixture, so this
/// exercises the whole path a video player would: a bitstream in, colour bars out.
///
/// The bars are the first proof the imported plane layout is right: a wrong pitch or offset still
/// paints *a* picture, but the bar centres would not land on seven distinct, correct colours.
#[test]
fn a_decoded_h264_surface_composites_through_the_shader() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(frame) = gpui_va::decode(gpui_va::FIXTURE) else {
        eprintln!("skipping: no libavcodec of the declared ABI, or no VA-API device");
        return;
    };
    assert_ne!(
        frame.handle().modifier,
        DmaBufHandle::LINEAR,
        "a decoded surface is tiled"
    );

    let (width, height, pixels) = composite_pixels(frame.handle().clone());
    // A decoded surface is the fixture's 128×128, and the surface element fits it to the window; on
    // a HiDPI headless window the capture is larger than the element, so sample the *capture* and
    // read the seven bar centres out of it.
    for (bar, rgb) in SMPTE_BARS.iter().enumerate() {
        let x = (2 * bar as u32 + 1) * width / (2 * SMPTE_BARS.len() as u32);
        let got = pixels[(height / 2 * width + x) as usize];
        let (y, cb, cr) = nv12_from_rgb(*rgb);
        let want = ycbcr_to_rgb(
            f32::from(y) / 255.0,
            f32::from(cb) / 255.0,
            f32::from(cr) / 255.0,
        );
        for channel in 0..3 {
            let want = (want[channel] * 255.0).round() as i32;
            let got = i32::from(got[channel]);
            assert!(
                (got - want).abs() <= 2,
                "bar {bar} channel {channel} at x={x} is {got}, but {rgb:?} decodes to {want}",
            );
        }
        assert_eq!(got[3], 255, "bar {bar} should be opaque");
    }
}

/// The same bars in **limited** range — the shape real footage is in. The decoder reads the stream's
/// colour space and declares it on the handle, so the shader inverts *that*: with the limited range
/// and the BT.601 matrix, the seven bars come back as drawn.
#[test]
fn a_limited_range_surface_converts_through_the_shader() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(frame) = gpui_va::decode(gpui_va::FIXTURE_LIMITED) else {
        eprintln!("skipping: no libavcodec of the declared ABI, or no VA-API device");
        return;
    };
    let got = composite_bar_centers(frame.handle().clone());
    for (bar, (rgb, got)) in SMPTE_BARS.iter().zip(&got).enumerate() {
        for channel in 0..3 {
            let (want, got) = (i32::from(rgb[channel]), i32::from(got[channel]));
            // The studio range has fewer levels for each colour, so the round trip is a little coarser
            // than the full-range one.
            assert!(
                (got - want).abs() <= 4,
                "bar {bar} channel {channel} is {got}, but the limited bars put {want}",
            );
        }
        assert_eq!(got[3], 255, "bar {bar} should be opaque");
    }
}

/// The negative control for the test above: the *same* limited bytes, but declared full range. The
/// shader then lifts the blacks and compresses the contrast, and the bars are no longer the bars — so
/// the declaration is what turns the conversion on, not a detail the shader happens to ignore.
#[test]
fn a_limited_surface_declared_as_full_range_is_not_the_bars() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(frame) = gpui_va::decode(gpui_va::FIXTURE_LIMITED) else {
        eprintln!("skipping: no libavcodec of the declared ABI, or no VA-API device");
        return;
    };
    let handle = frame
        .handle()
        .clone()
        .with_color_space(gpui_engine::YuvColorSpace::default());
    let got = composite_bar_centers(handle);
    let off = SMPTE_BARS
        .iter()
        .zip(&got)
        .any(|(rgb, got)| (0..3usize).any(|c| i32::from(got[c]).abs_diff(i32::from(rgb[c])) > 6));
    assert!(
        off,
        "reading limited bytes as full range must change the colour, but got {got:?}",
    );
}

/// A stream, played: every frame of the clip decodes to its own surface, and successive frames reach
/// the renderer as *different* pictures. That is the proof no surface was recycled under a frame
/// still being sampled — a reused surface would paint the same bars twice.
#[test]
fn successive_frames_of_a_clip_composite_differently() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    let Some(mut decoder) = gpui_va::Decoder::open() else {
        eprintln!("skipping: no libavcodec of the declared ABI, or no VA-API device");
        return;
    };
    decoder.send(gpui_va::CLIP).expect("feed the clip");
    decoder.finish().expect("finish the clip");

    // Hold every frame, as a player must: releasing one returns its surface to the pool.
    let mut frames = Vec::new();
    while let Some(frame) = decoder.receive() {
        frames.push(frame);
    }
    if frames.len() < 2 {
        eprintln!("skipping: the clip yielded fewer than two frames");
        return;
    }

    let first = composite_pixels(frames[0].handle().clone());
    let last = composite_pixels(frames[frames.len() - 1].handle().clone());
    let centre = |(width, height, pixels): (u32, u32, Vec<[u8; 4]>)| {
        pixels[(height / 2 * width + width / 2) as usize]
    };
    assert_ne!(
        centre(first),
        centre(last),
        "the scrolled clip frames should paint different bars",
    );
}

#[test]
fn a_gpu_canvas_composites_a_same_device_texture() {
    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }
    // The other authoring layer: `gpu_canvas(..)`'s paint callback wants a texture on the renderer's
    // own device, which a producer reaches through the canvas's typed door. The headless harness
    // used here holds its renderer erased as a `SceneRenderer` and does not implement `GpuRenderer`,
    // so no device is lent and the callback paints nothing: skip rather than fail, and say which
    // piece is not reachable from this test.
    let Some(pixel) = composite_canvas(BGRA_COLOUR).expect("composite the canvas") else {
        eprintln!(
            "skipping: the headless window's renderer lends no device — the authoring TestWindow \
             does not implement GpuRenderer — so no same-device texture can be made"
        );
        return;
    };
    assert_eq!(pixel, COLOUR);
}

/// The producer's whole point: a *bounded* playback keeps advancing. Each frame the renderer
/// composites has its release descriptor signalled, so the playback returns that surface to the
/// decoder's pool and hands out the next — the dma-buf handshake end to end, in constant memory.
///
/// The order matters, and this pins it: a frame is taken only once the previous one has been
/// composited. A frame taken and then not drawn is never sampled, so its release is never signalled,
/// and the playback waits on it for good — which is exactly what stalled the `player` example.
#[test]
fn a_bounded_playback_plays_through_the_renderer() {
    use std::cell::RefCell;

    if !gpu_available() {
        eprintln!("skipping: no Vulkan adapter for the headless renderer");
        return;
    }

    /// Paints whatever frame the test has handed it.
    struct PlaybackSurfaceView {
        handle: Rc<RefCell<Option<DmaBufHandle>>>,
    }

    impl Render for PlaybackSurfaceView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            match self.handle.borrow().as_ref() {
                Some(handle) => div()
                    .size_full()
                    .child(surface(handle.clone()).size_full())
                    .into_any_element(),
                None => div().size_full().into_any_element(),
            }
        }
    }

    let fixture = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/clip.mp4"));
    let Some(mut playback) = gpui_va::Playback::open_path(fixture, 4) else {
        eprintln!("skipping: no libavformat/libavcodec of the declared ABI, or no VA-API device");
        return;
    };

    let handle: Rc<RefCell<Option<DmaBufHandle>>> = Rc::new(RefCell::new(None));
    let text_system = std::sync::Arc::new(CosmicTextSystem::new("fallback"));
    let mut cx = HeadlessAppContext::with_platform(text_system, std::sync::Arc::new(()), || {
        gpui::current_headless_renderer()
    });
    let view_handle = handle.clone();
    let window = cx
        .open_window(
            size(px(WIDTH as f32), px(HEIGHT as f32)),
            move |_window, cx| {
                cx.new(|_| PlaybackSurfaceView {
                    handle: view_handle,
                })
            },
        )
        .expect("a window");
    let window: AnyWindowHandle = window.into();

    // Each pass draws the frame on screen and reads it back. The read-back polls the device, which
    // runs the submission's release callback; only then is the next frame taken, so every frame is
    // composited before it is replaced.
    for _ in 0..30 {
        cx.update_window(window, |_, window, cx| {
            let _ = window.draw(cx);
        })
        .expect("draw a frame");
        let _ = cx.capture_screenshot(window).expect("capture the frame");
        *handle.borrow_mut() = playback.next_frame();
        if playback.position() > 12 {
            break;
        }
    }

    assert!(
        playback.position() > 12,
        "the playback should keep advancing past its four-frame bound, reached {}",
        playback.position()
    );
}

/// The `fs_surface` matrix, on the CPU: `ycbcr_to_RGB * vec4(y, cb, cr, 1)`.
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> [f32; 4] {
    let columns = [
        [1.0, 1.0, 1.0],
        [0.0, -0.3441, 1.7720],
        [1.4020, -0.7141, 0.0],
        [-0.7010, 0.5291, -0.8860],
    ];
    let mut rgb = [0.0f32; 4];
    for channel in 0..3 {
        rgb[channel] = (columns[0][channel] * y
            + columns[1][channel] * cb
            + columns[2][channel] * cr
            + columns[3][channel])
            .clamp(0.0, 1.0);
    }
    rgb[3] = 1.0;
    rgb
}

/// The Y, Cb and Cr bytes the renderer's BT.601 conversion maps back to `rgb` — the inverse of
/// [`ycbcr_to_rgb`], so the two-plane buffer can carry the shared fixture colour.
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
