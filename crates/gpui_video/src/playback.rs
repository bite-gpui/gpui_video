//! The facade's own player: a source opened, read and decoded on a dedicated thread, whose decoded
//! frames are handed to the UI thread — and only the frames.
//!
//! This is what makes a player *optional*: an application that just wants to play a file writes
//! `video().source("movie.mp4")`, and the facade does the rest, through the platform crate it was
//! compiled with (`gpui_va` on Linux today, `gpui_cv`/`gpui_mf` as they arrive). An application that
//! already has a player of its own passes that instead, through [`Video::player`](crate::Video::player).
//!
//! # Why a thread
//!
//! Opening a container, reading its bytes and driving the decoder are all CPU work on the calling
//! thread — the GPU does the pixels, not the pipeline — and reading or decoding can block. Done on
//! the UI thread, every one of those stalls a frame. So the whole pipeline lives on a dedicated
//! thread, and the UI thread only *presents*: it asks for a frame when one is due and the window can
//! show it, and composites whatever comes back.
//!
//! # What crosses, and what does not
//!
//! The pipeline can't cross threads — ffmpeg and VA-API contexts hold raw pointers and are not
//! `Send` — so nothing of it does; it is opened, used and dropped on one thread. What crosses is the
//! **decoded frame**: a [`SurfaceSource`], which on this path is a dma-buf — an fd and a layout, both
//! `Send`. The dma-buf *is* the handoff: a kernel object the GPU samples regardless of which CPU
//! thread exported it, so no bytes are copied and no arena is needed. The frame's release descriptor
//! (an `eventfd` the renderer signals) is what lets the pool recycle a surface across the thread
//! boundary without the two ever sharing memory.

use std::time::{Duration, Instant};

use gpui_authoring::{Context, SurfaceSource};
use gpui_video_platform::SurfaceProducer;

use crate::Player;

/// How many frames the facade's player keeps alive at once. Small, because recycling is what allows
/// a stream to play in constant memory.
const DEPTH: usize = 4;

/// The frame period for a source that declares no rate of its own — 25 fps, the bundled clips' rate.
const DEFAULT_PERIOD: Duration = Duration::from_millis(40);

/// A video source: a file to open, or a stream already in memory.
///
/// The `From` impls are what let [`Video::source`](crate::Video::source) take `"movie.mp4"`, a
/// `String`, or a byte slice (`gpui_va::CLIP`, say).
#[derive(Clone, Debug)]
pub enum Source {
    /// A path opened through the platform's demuxer: a container (MP4, Matroska, …) or a raw
    /// elementary stream (Annex B).
    Path(String),
    /// An elementary stream already in memory.
    Stream(Vec<u8>),
}

impl From<&str> for Source {
    fn from(path: &str) -> Self {
        Source::Path(path.to_owned())
    }
}

impl From<String> for Source {
    fn from(path: String) -> Self {
        Source::Path(path)
    }
}

impl From<&[u8]> for Source {
    fn from(bytes: &[u8]) -> Self {
        Source::Stream(bytes.to_vec())
    }
}

impl From<Vec<u8>> for Source {
    fn from(bytes: Vec<u8>) -> Self {
        Source::Stream(bytes)
    }
}

/// What a platform backend gives the facade: a producer that also knows its own frame rate.
///
/// `gpui_video` is the facade, so it names the backend it was built with; the backend cannot name
/// it back (that would be a cycle), so the trait lives here and is implemented for the backend's
/// concrete playback type.
trait BackendPlayback: SurfaceProducer {
    fn frame_period(&self) -> Option<Duration>;
    fn duration(&self) -> Option<Duration>;
    fn is_seekable(&self) -> bool;
    fn seek(&mut self, to: Duration) -> bool;
}

#[cfg(target_os = "linux")]
impl BackendPlayback for gpui_va::Playback {
    fn frame_period(&self) -> Option<Duration> {
        gpui_va::Playback::frame_period(self)
    }

    fn duration(&self) -> Option<Duration> {
        gpui_va::Playback::duration(self)
    }

    fn is_seekable(&self) -> bool {
        gpui_va::Playback::is_seekable(self)
    }

    fn seek(&mut self, to: Duration) -> bool {
        gpui_va::Playback::seek(self, to)
    }
}

/// Open `source` with the platform's decoder, or `None` where that backend is not compiled in, its
/// libraries are missing, or the source yields nothing.
#[cfg(target_os = "linux")]
fn open_backend(source: &Source) -> Option<Box<dyn BackendPlayback>> {
    let playback = match source {
        Source::Path(path) => gpui_va::Playback::open_path(std::path::Path::new(path), DEPTH),
        Source::Stream(bytes) => gpui_va::Playback::open(bytes, DEPTH),
    };
    playback.map(|playback| Box::new(playback) as Box<dyn BackendPlayback>)
}

#[cfg(not(target_os = "linux"))]
fn open_backend(_source: &Source) -> Option<Box<dyn BackendPlayback>> {
    None
}

/// What the UI thread tells the service to do.
enum Command {
    /// The window can (or can no longer) show a frame — pacing needs this, because a frame taken for
    /// a window that cannot show it is never released.
    Visible(bool),
    /// Advance one frame, if the service has one to give.
    Advance,
    /// Move playback to a position.
    Seek(Duration),
}

/// What the service tells the UI thread.
#[cfg(target_os = "linux")]
enum Event {
    /// The backend opened, with its frame period and — where the source declares one — its length,
    /// and whether it can be seeked.
    /// `period` is `None` where it failed to open; `duration` where the source has no length.
    Opened {
        period: Option<Duration>,
        duration: Option<Duration>,
        seekable: bool,
    },
    /// A decoded frame to present.
    Frame(SurfaceSource),
}

/// The service thread's whole life: open the source, then serve commands until the UI thread drops
/// its end. The pipeline is created, used and dropped here, so it never has to be `Send`.
#[cfg(target_os = "linux")]
fn serve(source: Source, commands: flume::Receiver<Command>, events: flume::Sender<Event>) {
    let mut playback = open_backend(&source);
    let period = playback
        .as_ref()
        .and_then(|playback| playback.frame_period())
        .unwrap_or(DEFAULT_PERIOD);
    let duration = playback.as_ref().and_then(|playback| playback.duration());
    let seekable = playback.as_ref().is_some_and(|playback| playback.is_seekable());
    if playback.is_none() {
        eprintln!("gpui_video: could not open {source:?}");
    }
    if events
        .send(Event::Opened {
            period: playback.is_some().then_some(period),
            duration,
            seekable,
        })
        .is_err()
    {
        return;
    }

    // The UI thread drives pacing (it is the one that knows the window's visibility and the clock),
    // so the service only ever produces on demand.
    let mut visible = false;
    while let Ok(command) = commands.recv() {
        match command {
            Command::Visible(value) => visible = value,
            // A stale `Advance` (the window hid between asking and now) produces nothing.
            Command::Advance if visible => {
                if let Some(playback) = &mut playback
                    && let Some(frame) = playback.surface()
                    && events.send(Event::Frame(frame)).is_err()
                {
                    break;
                }
            }
            Command::Advance => {}
            Command::Seek(to) => {
                if let Some(playback) = &mut playback {
                    playback.seek(to);
                }
            }
        }
    }
}

/// The player a `.source(...)` video runs on: a thin face on the service thread.
///
/// It holds no pipeline — only the frame most recently handed over, the channel to the service, and
/// the pacing state the view drives with [`tick`](OwnedPlayer::tick), which is where the *visibility*
/// comes from: the service must not produce a frame for a window that cannot show it, because a frame
/// taken and never composited is never released.
pub(crate) struct OwnedPlayer {
    /// The service's inbox; `None` where no backend exists for this target.
    commands: Option<flume::Sender<Command>>,
    /// Whether the window is visible, so a change is only sent when it changes.
    visible: bool,
    /// Whether the backend opened; nothing plays until it has.
    opened: bool,
    playing: bool,
    /// Whether a requested frame is still in flight, so at most one is ever outstanding.
    pending: bool,
    /// Set by a seek, cleared when the frame at the new position arrives: it forces one frame even
    /// while paused, so a seek is seen.
    seek_pending: bool,
    /// The latest position a seek asked for while another was still resolving. A drag reports a
    /// position per pointer move, and each is a decode-forward on the service thread, so only the
    /// newest is kept and sent once the one in flight lands — never a queue of them.
    queued_seek: Option<Duration>,
    /// Whether the source can be seeked.
    seekable: bool,
    /// Set while a scrub is in progress: whether playback was running when it began, so it can
    /// resume — or stay paused — when the scrub ends.
    scrub_was_playing: Option<bool>,
    /// When the next frame comes due, so playback keeps the stream's rate.
    due: Instant,
    period: Duration,
    /// Playback time advanced so far, for the timecode.
    elapsed: Duration,
    /// How long the source is, where it declares a length.
    duration: Option<Duration>,
    /// The frame on screen: the scene re-samples it until the next arrives.
    current: Option<SurfaceSource>,
    /// Whether to trace frames to stderr (`GPUI_VIDEO_TRACE`), for watching the pipeline from a
    /// terminal when the picture is on another display.
    trace: bool,
}

impl OwnedPlayer {
    /// Start the service thread for `source`, and return a player that shows nothing until a frame
    /// arrives.
    ///
    /// The open — demuxing the file, bringing up the decoder — happens on the service thread, so the
    /// frame that creates the player never blocks on it.
    pub(crate) fn new(source: Source, cx: &mut Context<Self>) -> Self {
        #[cfg(target_os = "linux")]
        let commands = start_service(source, cx);
        #[cfg(not(target_os = "linux"))]
        let commands = {
            let _ = (source, cx);
            None
        };

        Self {
            commands,
            visible: false,
            opened: false,
            playing: true,
            pending: false,
            seek_pending: false,
            queued_seek: None,
            seekable: false,
            scrub_was_playing: None,
            due: Instant::now(),
            period: DEFAULT_PERIOD,
            elapsed: Duration::ZERO,
            duration: None,
            current: None,
            trace: std::env::var_os("GPUI_VIDEO_TRACE").is_some(),
        }
    }

    /// Take an event from the service: the open, or a frame to present.
    #[cfg(target_os = "linux")]
    fn on_event(&mut self, event: Event) {
        match event {
            Event::Opened {
                period,
                duration,
                seekable,
            } => {
                if self.trace {
                    eprintln!(
                        "gpui_video: opened, period {period:?}, duration {duration:?}, seekable {seekable}"
                    );
                }
                self.opened = period.is_some();
                if let Some(period) = period {
                    self.period = period;
                }
                self.duration = duration;
                self.seekable = seekable;
                self.due = Instant::now();
            }
            Event::Frame(frame) => {
                if self.trace {
                    eprintln!("gpui_video: frame at {:.2}s", self.elapsed.as_secs_f64());
                }
                self.current = Some(frame);
                self.pending = false;
                self.seek_pending = false;
                self.elapsed += self.period;
                // A looping source restarts when the file ends, so the timecode wraps rather than
                // running past the length it declared.
                if let Some(duration) = self.duration
                    && self.elapsed >= duration
                {
                    self.elapsed = Duration::ZERO;
                }
                // A drag's newest position, asked for while this seek was still resolving: send it
                // now, so a scrub follows the pointer without queueing a decode per move.
                if let Some(to) = self.queued_seek.take() {
                    self.elapsed = to;
                    self.pending = false;
                    self.seek_pending = true;
                    self.send(Command::Seek(to));
                }
            }
        }
    }

    /// Advance at most one frame, only when a frame is due and the window can show it. The wall clock
    /// keeps the stream's rate; a request never stacks, so at most one frame is ever outstanding, and
    /// a run-ahead — the window stalled — is resynced rather than fast-forwarded through frames that
    /// would then never be shown.
    pub(crate) fn tick(&mut self, visible: bool) {
        if visible != self.visible {
            self.visible = visible;
            self.send(Command::Visible(visible));
        }
        if !visible || !self.opened || self.pending {
            return;
        }
        // A seek shows its frame at once, even while paused; ordinary advancement waits for the
        // frame to come due.
        if self.seek_pending {
            if self.send(Command::Advance) {
                self.pending = true;
            }
            return;
        }
        if !self.playing {
            return;
        }
        let now = Instant::now();
        if now.duration_since(self.due) >= self.period {
            // Move the deadline on by exactly one period, so playback keeps the stream's own rate
            // across render ticks rather than the render loop's. Setting the deadline to `now` here
            // would discard the remainder every tick and, on a render loop only a little faster than
            // the stream, halve the rate. If the deadline has fallen a whole period behind the clock,
            // the window stalled — resync to `now` rather than burst through frames that would never
            // be shown.
            self.due += self.period;
            if now.duration_since(self.due) >= self.period {
                self.due = now;
            }
            if self.send(Command::Advance) {
                self.pending = true;
            }
        }
    }

    /// Send a command to the service, if there is one. `false` where the service is gone.
    fn send(&self, command: Command) -> bool {
        self.commands
            .as_ref()
            .is_some_and(|commands| commands.send(command).is_ok())
    }

    /// Note a transport transition when tracing is on, so the state machine is visible from a
    /// terminal while the picture is on another display.
    fn note(&self, message: impl std::fmt::Display) {
        if self.trace {
            eprintln!("gpui_video: {message}");
        }
    }
}

/// Start the service thread and the task that drains its events, returning the service's inbox.
///
/// The service runs on its own OS thread, because the pipeline it owns is not `Send` and cannot live
/// in the background pool, where a task may be polled by any worker: one thread opens, reads and
/// decodes, and nothing of that ever leaves it.
#[cfg(target_os = "linux")]
fn start_service(
    source: Source,
    cx: &mut Context<OwnedPlayer>,
) -> Option<flume::Sender<Command>> {
    let (commands_tx, commands_rx) = flume::unbounded::<Command>();
    let (events_tx, events_rx) = flume::unbounded::<Event>();

    let spawned = std::thread::Builder::new()
        .name("gpui-video".to_owned())
        .spawn(move || serve(source, commands_rx, events_tx));
    if spawned.is_err() {
        return None;
    }

    // The drain task is foreground: `entity.update` is the bus back into the reactive graph, and the
    // `notify` is what wakes the frame the arriving frame is painted in.
    cx.spawn(async move |this, cx| {
        while let Ok(event) = events_rx.recv_async().await {
            if this
                .update(cx, |player, cx| {
                    player.on_event(event);
                    cx.notify();
                })
                .is_err()
            {
                break;
            }
        }
    })
    .detach();

    Some(commands_tx)
}

impl Player for OwnedPlayer {
    fn surface(&mut self) -> Option<SurfaceSource> {
        self.current.clone()
    }

    fn is_playing(&self) -> bool {
        // Nothing is playing until the backend has opened, so the view holds its animation frames
        // until there is something to show.
        self.playing && self.opened
    }

    fn play(&mut self) {
        self.playing = true;
        self.due = Instant::now();
        self.note("play");
    }

    fn pause(&mut self) {
        self.playing = false;
        self.note("pause");
    }

    fn toggle(&mut self) {
        // An explicit play/pause supersedes a scrub: a pointer release the window never saw can leave
        // a scrub latched, and its stray resume must not fight the button.
        self.scrub_was_playing = None;
        if self.is_playing() {
            self.pause();
        } else {
            self.play();
        }
    }

    fn is_seekable(&self) -> bool {
        self.seekable
    }

    fn seek(&mut self, to: Duration) {
        if !self.seekable {
            return;
        }
        // Show the position at once, and ask for the frame at it.
        self.elapsed = to;
        self.due = Instant::now();
        if self.seek_pending {
            // A seek is still resolving. A drag reports many positions, and each is a decode-forward
            // on the service thread, so keep only the latest: it is sent when the one in flight
            // lands, rather than queueing a decode per pointer move.
            self.note(format_args!("seek to {to:?} (coalesced)"));
            self.queued_seek = Some(to);
            return;
        }
        // Drop any frame in flight (it is at the old position) and ask for the new one.
        self.pending = false;
        self.seek_pending = true;
        self.note(format_args!("seek to {to:?}"));
        self.send(Command::Seek(to));
    }

    fn scrub_started(&mut self) {
        self.note(format_args!("scrub_started (playing={})", self.playing));
        self.scrub_was_playing = Some(self.playing);
        if self.playing {
            self.pause();
        }
    }

    fn scrub_ended(&mut self) {
        self.note(format_args!("scrub_ended (was_playing={:?})", self.scrub_was_playing));
        if self.scrub_was_playing.take() == Some(true) {
            self.play();
        }
    }

    fn scrubbing(&self) -> bool {
        self.scrub_was_playing.is_some()
    }

    fn position(&self) -> Duration {
        self.elapsed
    }

    fn duration(&self) -> Option<Duration> {
        self.duration
    }
}
