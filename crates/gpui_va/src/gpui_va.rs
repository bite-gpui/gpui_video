//! A Linux producer for the surface path: a VA-API decode surface, exported as a dma-buf.
//!
//! The escape hatch lets a producer hand the renderer pixels gpui did not draw, and on Linux the
//! currency is a dma-buf. The shape a video pipeline actually produces — and the one this crate
//! stands in for — is a hardware-decoded `NV12` surface: the VA driver allocates it, a decoder writes
//! it, and `vaExportSurfaceHandle` hands out the object the renderer imports.
//!
//! A decoded surface is **tiled** (here `I915_FORMAT_MOD_Y_TILED`), which the renderer accepts only
//! because its device enables `VK_EXT_image_drm_format_modifier`; see `gpui_wgpu`'s device creation.
//! The surface's memory is not the CPU's to write — iHD maps none of it and refuses `vaPutImage` with
//! `VA_STATUS_ERROR_SURFACE_BUSY` — so the fixture is filled on the GPU, by importing the exported
//! object into a Vulkan device and copying into it, which is what a decoder would do with real
//! bitstream.
//!
//! `libva` is opened at run time, so a machine without the driver builds and skips.

#![cfg(target_os = "linux")]

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

/// The version gate for `libavcodec`/`libavutil`, and where their symbols will be resolved.
///
/// ffmpeg bumps a library's soname exactly when its ABI changes, so `libavcodec.so.62` *is* the ABI
/// version — and the versioned `VkImage`-style declarations this crate will carry are written against
/// it. Loading that soname, then resolving a symbol *by version tag* (`LIBAVCODEC_62`), refuses a
/// library that is present but not the ABI declared for, rather than mis-reading its structs.
mod ffmpeg {
    use std::ffi::{CString, c_char, c_int, c_void};

    /// The `(libavcodec, libavutil)` majors the decode declarations are written against.
    ///
    /// libavcodec 62 / libavutil 60 is ffmpeg 8.0.
    const ABI: (u32, u32) = (62, 60);

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

    /// A loaded, ABI-checked `libavcodec` and `libavutil`, held for as long as the decode uses them.
    #[allow(dead_code, reason = "the decode wiring resolves symbols through it next")]
    pub struct Ffmpeg {
        codec: *mut c_void,
        util: *mut c_void,
    }

    impl Drop for Ffmpeg {
        fn drop(&mut self) {
            unsafe {
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
    #[allow(dead_code, reason = "the decode wiring calls it next")]
    pub fn load() -> Option<Ffmpeg> {
        let (codec_major, util_major) = ABI;
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
        Some(Ffmpeg { codec, util })
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
    }
}

/// The VA-API FFI: create an NV12 surface with the driver, and export it as a dma-buf.
mod va {
    #![allow(non_upper_case_globals)]
    #![allow(unsafe_op_in_unsafe_fn)]

    use std::ffi::c_void;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    use gpui_engine::{DmaBufFormat, DmaBufHandle, DmaBufPlane};
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
        let export: ExportSurfaceHandle = unsafe { *va.get(b"vaExportSurfaceHandle\0")? };

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

        let mut descriptor = RmDescriptor::default();
        anyhow::ensure!(
            export(
                display,
                surface,
                VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                &mut descriptor,
            ) == 0,
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
        let handle = DmaBufHandle::new(
            descriptor.width,
            descriptor.height,
            DmaBufFormat::Nv12,
            object.drm_format_modifier,
            [
                DmaBufPlane::new(fd, u64::from(luma.offset[0]), luma.pitch[0]),
                DmaBufPlane::new(chroma_fd, u64::from(chroma.offset[0]), chroma.pitch[0]),
            ],
            None,
        );
        Ok((producer, handle))
    }
}
