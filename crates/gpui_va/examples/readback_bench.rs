//! How the two ways a decoded frame reaches the screen compare.
//!
//! A player either hands the renderer the surface the decoder wrote — a dma-buf it imports and samples
//! on the GPU, no copy through the CPU — or reads the pixels back and hands over a packed image. This
//! measures both on the same file, and splits the readback into its GPU→CPU transfer and its software
//! conversion so it is clear which part costs — and, with the staged pass, whether the transferred
//! buffer itself is slow to read.
//!
//! ```sh
//! cargo run --release -p gpui_va --example readback_bench -- movie.mp4 [frames]
//! ```
//!
//! The figures are the *producer* side. They do not include the renderer's per-frame import of the
//! dma-buf, or its upload of the readback pixels (a CPU→GPU copy, on the UI thread).

use std::path::Path;
use std::time::{Duration, Instant};

use gpui_va::Demuxer;

/// How many frames to measure when no count is given.
const FRAMES: u32 = 300;

/// Which readback to measure.
enum Mode {
    /// Fresh software frame and output buffer per frame (`to_bgra_timed`).
    Allocating,
    /// Software frame and output buffer reused across frames (`Readback::read`).
    Reusing,
    /// Reused, and the planes copied into a cached staging buffer before converting.
    Staged,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: readback_bench <file> [frames]");
        return;
    };
    let frames: u32 = args.next().and_then(|n| n.parse().ok()).unwrap_or(FRAMES);
    let path = Path::new(&path);

    let Some((plain, size)) = decode(path, frames) else {
        eprintln!("could not open: libavformat/libavcodec/libva of the declared ABI, or no video");
        return;
    };
    let Some(allocating) = readback(path, frames, Mode::Allocating) else {
        eprintln!("could not read back");
        return;
    };
    let Some(reusing) = readback(path, frames, Mode::Reusing) else {
        return;
    };
    let Some(staged) = readback(path, frames, Mode::Staged) else {
        return;
    };

    let (width, height) = size;
    let ms = |elapsed: Duration| elapsed.as_secs_f64() * 1000.0 / f64::from(frames);
    let fps = |elapsed: Duration| f64::from(frames) / elapsed.as_secs_f64();

    println!("{width}x{height}, {frames} frames measured");
    println!(
        "decode + export   (zero-copy producer): {:>7.3} ms/frame  {:>7.0} fps",
        ms(plain),
        fps(plain),
    );
    for (label, run) in [
        ("allocating", &allocating),
        ("reusing", &reusing),
        ("reusing + staged copy", &staged),
    ] {
        println!(
            "decode + readback ({label:>21}): {:>7.3} ms/frame  {:>7.0} fps",
            ms(run.total),
            fps(run.total),
        );
        println!(
            "  GPU→CPU transfer: {:>6.3} ms   staging copy: {:>6.3} ms   NV12→BGRA: {:>6.3} ms",
            ms(run.transfer),
            ms(run.copy),
            ms(run.convert),
        );
    }
    let bytes = u64::from(width) * u64::from(height) * 4;
    println!(
        "each read-back frame is {:.2} MiB, which the readback path also uploads to the GPU",
        bytes as f64 / (1024.0 * 1024.0),
    );
}

/// A readback pass's totals.
struct Readback {
    total: Duration,
    transfer: Duration,
    copy: Duration,
    convert: Duration,
}

/// Decode up to `frames` frames, their surfaces recycled, and time the lot — the zero-copy producer.
fn decode(path: &Path, frames: u32) -> Option<(Duration, (u32, u32))> {
    let mut demuxer = Demuxer::open(path)?;
    let mut decoder = demuxer.decoder()?;
    let mut size = (0, 0);
    let mut decoded = 0;

    let start = Instant::now();
    while demuxer.send_next(&mut decoder).ok()? {
        while let Some(frame) = decoder.receive() {
            let handle = frame.handle();
            size = (handle.width as u32, handle.height as u32);
            decoded += 1;
            if decoded >= frames {
                return Some((start.elapsed(), size));
            }
        }
    }
    (decoded > 0).then(|| (start.elapsed(), size))
}

/// Decode and read each frame back to the CPU in the given `mode`, reporting the parts.
fn readback(path: &Path, frames: u32, mode: Mode) -> Option<Readback> {
    let mut demuxer = Demuxer::open(path)?;
    let mut decoder = demuxer.decoder()?;
    let mut decoded = 0;
    let (mut transfer, mut copy, mut convert) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut scratch: Option<gpui_va::Readback> = None;
    let mut staging: Vec<u8> = Vec::new();

    let start = Instant::now();
    while demuxer.send_next(&mut decoder).ok()? {
        while let Some(frame) = decoder.receive() {
            match mode {
                Mode::Allocating => {
                    if let Some((_, moved, converted)) = frame.to_bgra_timed() {
                        transfer += moved;
                        convert += converted;
                    }
                }
                Mode::Reusing => {
                    let scratch = scratch.get_or_insert_with(|| frame.readback().unwrap());
                    if let Some(result) = scratch.read(&frame) {
                        transfer += result.transfer;
                        convert += result.convert;
                    }
                }
                Mode::Staged => {
                    let scratch = scratch.get_or_insert_with(|| frame.readback().unwrap());
                    if let Some(result) = scratch.read_staged(&frame, &mut staging) {
                        transfer += result.transfer;
                        copy += result.copy;
                        convert += result.convert;
                    }
                }
            }
            decoded += 1;
            if decoded >= frames {
                return Some(Readback {
                    total: start.elapsed(),
                    transfer,
                    copy,
                    convert,
                });
            }
        }
    }
    (decoded > 0).then(|| Readback {
        total: start.elapsed(),
        transfer,
        copy,
        convert,
    })
}
