//! The video view: whatever a player hands over, composited every frame, with HTML5-shaped controls.
//!
//! `gpui_video` is the facade over the platform producers — `gpui_va` on Linux today, `gpui_cv` and
//! `gpui_mf` as they arrive — so an application never names a decoder. It either gives the view a
//! source and lets the facade open one, or hands it a player entity of its own.
//!
//! ```
//! # use gpui_video::video;
//! // No player to manage: the facade opens the decoder through the backend it was built with, and
//! // drives it — pacing, transport bar and all.
//! let picture = video().source("movie.mp4").controls(true);
//! # let _ = picture;
//! ```
//!
//! ```no_run
//! # use gpui_authoring::{Entity, IntoElement};
//! # use gpui_video::{Player, video};
//! # fn make<P: Player>(player: Entity<P>) -> impl IntoElement {
//! // Or an entity owns the player, and the view is a face on it: `cx.notify()` on the player
//! // repaints the view, and the default controls read that same entity.
//! video().player(player).controls(true)
//! # }
//! ```
//!
//! Controls are headless and overridable, and drawn over the frame. The headless state is
//! [`Transport`]; [`video_controls`] is one rendering of it (an HTML5-shaped bar), and
//! [`Video::controls`] takes `true`, `false`, or any element — so a user-defined bar replaces the
//! default by being passed in its place.
//!
//! Pacing belongs to whoever owns the frames. The [`source`](Video::source) shape's player is the
//! facade's own, and the view drives it — the view is where the window's visibility lives, and a
//! frame taken for a window that cannot show it would never be released. A bare
//! [`SurfaceProducer`] handed in with [`producer`](Video::producer) has no cadence of its own, so the
//! view advances it once per render.

use std::time::Duration;

use gpui_authoring::{
    AnyElement, App, AppContext as _, ElementId, Entity, EntityId, IntoElement, ObjectFit,
    ParentElement, StyleRefinement, Styled, SurfaceSource, View, ViewElement, Window, div, surface,
};
use gpui_video_platform::SurfaceProducer;

mod controls;
mod playback;

pub use controls::{Controls, Transport, VideoControls, video_controls};
pub use playback::Source;

use playback::OwnedPlayer;

/// The player side of a video, for a player that lives in an entity.
///
/// [`Video::player`] accepts an [`Entity`] whose type implements this, so the view is generic over
/// the player rather than over a concrete decoder: an application builds the player its platform
/// has, keeps it in an entity, and the view reads the current frame from it.
///
/// Every [`SurfaceProducer`] is a `Player` for free, and the transport below is all optional — a
/// producer with no transport is "always playing", with no position to report and nothing to seek.
/// A real player overrides the methods it can answer; the surface seam itself stays surface-only.
pub trait Player: 'static {
    /// The surface to composite this frame, or `None` to keep showing the last one.
    fn surface(&mut self) -> Option<SurfaceSource>;

    /// Whether playback is running. A player with no transport is always running.
    fn is_playing(&self) -> bool {
        true
    }

    /// Begin playback.
    fn play(&mut self) {}

    /// Halt playback, keeping the current frame.
    fn pause(&mut self) {}

    /// Play if paused, pause if playing.
    fn toggle(&mut self) {
        if self.is_playing() {
            self.pause();
        } else {
            self.play();
        }
    }

    /// Where playback is, as the stream reports it. Zero for a player with no position.
    fn position(&self) -> Duration {
        Duration::ZERO
    }

    /// How long the stream is, when it knows. `None` for a live or looping stream with no length.
    fn duration(&self) -> Option<Duration> {
        None
    }

    /// Whether [`seek`](Player::seek) does anything. A player must opt in.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Move playback to `to`. A no-op unless [`is_seekable`](Player::is_seekable).
    fn seek(&mut self, _to: Duration) {}

    /// A scrub began on a controls rail. A player that can pause should stop for the duration, so the
    /// scrub is not fought by playback running underneath it.
    fn scrub_started(&mut self) {}

    /// The scrub ended. A player that paused at [`scrub_started`](Player::scrub_started) resumes.
    fn scrub_ended(&mut self) {}

    /// Whether a scrub is in progress — between [`scrub_started`](Player::scrub_started) and
    /// [`scrub_ended`](Player::scrub_ended). A controls rail uses it to ignore pointer drags it did
    /// not start.
    fn scrubbing(&self) -> bool {
        false
    }
}

impl<T: SurfaceProducer + 'static> Player for T {
    fn surface(&mut self) -> Option<SurfaceSource> {
        SurfaceProducer::surface(self)
    }
}

/// A video view: the current frame of a player, composited every frame.
///
/// Built by [`video`] and given frames with [`player`](Video::player) or [`producer`](Video::producer).
pub struct Video {
    frames: Frames,
    identity: Option<EntityId>,
    /// The id the view-owned shape keys its element state under: the `video()` call site, unless an
    /// entity gives the view its identity instead.
    id: ElementId,
    /// Builds the default controls, when there is a player entity to read them from.
    default_controls: Option<Box<dyn Fn() -> AnyElement>>,
    /// A free-placed overlay, from [`Video::overlay`]. Takes precedence over [`controls`](Video::controls).
    overlay: Option<AnyElement>,
    /// The controls mode, from [`Video::controls`].
    controls: Option<Controls>,
    object_fit: ObjectFit,
    style: StyleRefinement,
}

/// Where a video's frames come from, and so where its state lives.
enum Frames {
    /// No frames yet: `video()` before it is told where to read from.
    None,
    /// Frames come from an external player entity, read every frame. The entity holds the decoder —
    /// and its position, pool and cadence — so the view itself keeps nothing.
    External(Box<dyn FnMut(&mut App) -> Option<SurfaceSource>>),
    /// Frames come from a producer the view owns: built once, on the first frame, and kept in element
    /// state under the id the view pushes while rendering.
    Owned(Box<dyn FnOnce() -> Box<dyn SurfaceProducer>>),
    /// Frames come from a source the view opens itself, through the platform backend it was built
    /// with, into a player it owns.
    Source(Source),
}

/// Begin a video. Give it frames with [`source`](Video::source), [`player`](Video::player), or
/// [`producer`](Video::producer), and controls with [`controls`](Video::controls).
///
/// The call site names the view: the view-owned shape keys its element state under it, so a `video()`
/// rendered from the same place every frame keeps its producer across frames. Give it a distinct call
/// site for each video on screen — the same caveat as `use_state`.
#[track_caller]
pub fn video() -> Video {
    Video {
        frames: Frames::None,
        identity: None,
        id: ElementId::CodeLocation(*core::panic::Location::caller()),
        default_controls: None,
        overlay: None,
        controls: None,
        object_fit: ObjectFit::Contain,
        style: StyleRefinement::default(),
    }
}

impl Video {
    /// Read frames from an external player entity.
    ///
    /// The entity owns the player and its state; the view carries the entity's identity, so notifying
    /// the entity repaints the view, and the default controls ([`controls(true)`](Video::controls))
    /// read that same entity.
    pub fn player<P: Player>(mut self, player: Entity<P>) -> Self {
        self.identity = Some(player.entity_id());
        let read: Box<dyn FnMut(&mut App) -> Option<SurfaceSource>> = {
            let player = player.clone();
            Box::new(move |cx: &mut App| player.update(cx, |player, _| player.surface()))
        };
        self.frames = Frames::External(read);
        self.default_controls = Some(Box::new(move || {
            video_controls(player.clone()).into_any_element()
        }));
        self
    }

    /// Own the producer: `open` builds it once, on the first frame, and the view keeps it in element
    /// state under its call-site id.
    ///
    /// The element tree is rebuilt every frame, so `open` is *called* every frame but *used* only
    /// once: a decoder opened here survives across frames, unlike one built eagerly at each render.
    /// There is no entity behind this shape, so the default controls have nothing to read; use
    /// [`source`](Video::source) when you want them.
    pub fn producer(mut self, open: impl FnOnce() -> Box<dyn SurfaceProducer> + 'static) -> Self {
        self.frames = Frames::Owned(Box::new(open));
        self
    }

    /// Open `source` through the platform's decoder, and play it.
    ///
    /// This is the no-entity shape: the view opens the decoder itself — `gpui_va` on Linux today,
    /// `gpui_cv`/`gpui_mf` as they arrive — on a background thread of its own, and keeps the player in
    /// an entity it creates on its first frame, so the default controls have something to read and
    /// drive. `source` takes `"movie.mp4"`, a `String`, or a byte slice such as `gpui_va::CLIP`.
    ///
    /// Opening, reading and decoding all happen on that thread, so none of it stalls the UI; only
    /// decoded frames cross back, and the frame that creates the player shows nothing until one
    /// arrives.
    pub fn source(mut self, source: impl Into<Source>) -> Self {
        self.frames = Frames::Source(source.into());
        self
    }

    /// Draw `controls` over the frame, pinned across its bottom edge — the HTML5 place.
    ///
    /// [`true`](Controls) draws the default bar, [`false`](Controls) draws nothing, and an element —
    /// [`video_controls(player)`](video_controls) or your own — is drawn in its place. That last case
    /// is what makes controls overridable: the default is just one element among any others.
    pub fn controls(mut self, controls: impl Into<Controls>) -> Self {
        self.controls = Some(controls.into());
        self
    }

    /// Draw `overlay` over the frame, filling its bounds — free placement, for anything that is not a
    /// bottom bar. Takes precedence over [`controls`](Video::controls).
    ///
    /// The overlay is a later sibling of the picture, so it is hit-tested first: what is drawn over
    /// the frame receives the pointer, not the surface under it.
    pub fn overlay(mut self, overlay: impl IntoElement) -> Self {
        self.overlay = Some(overlay.into_any_element());
        self
    }

    /// Sets how the frame is fitted into the view's bounds.
    ///
    /// The frame carries its own pixel size and the view carries its layout size; the fit decides how
    /// the former maps onto the latter, using the same [`ObjectFit`] modes images use.
    /// [`ObjectFit::Contain`] (the default) letterboxes the frame inside the bounds while preserving
    /// its aspect ratio; [`ObjectFit::Fill`] stretches it to fill the bounds exactly.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }
}

impl Styled for Video {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl IntoElement for Video {
    type Element = ViewElement<Video>;

    fn into_element(self) -> Self::Element {
        ViewElement::new(self)
    }
}

impl View for Video {
    fn entity_id(&self) -> Option<EntityId> {
        self.identity
    }

    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let Video {
            frames,
            identity: _,
            id,
            default_controls,
            overlay,
            controls,
            object_fit,
            style,
        } = self;

        let wants_default_controls = matches!(controls, Some(Controls::Default));

        let (source, default_bar) = match frames {
            Frames::None => (None, None),
            Frames::External(mut read) => (read(cx), default_controls.map(|build| build())),
            Frames::Owned(open) => (owned_source(window, id, open), None),
            Frames::Source(source) => owned_player(window, cx, id, source, wants_default_controls),
        };

        let overlay = match overlay {
            Some(overlay) => Some(fill(overlay)),
            None => match controls {
                Some(Controls::None) | None => None,
                Some(Controls::Custom(element)) => Some(pin(element)),
                Some(Controls::Default) => default_bar.map(pin),
            },
        };

        match overlay {
            // The style lands on the container, and the frame fills it, so the overlay can be
            // positioned against the picture's own bounds.
            Some(overlay) => {
                let mut container = div().relative();
                *container.style() = style;
                container
                    .child(frame_element(source, object_fit, None))
                    .child(overlay)
                    .into_any_element()
            }
            None => frame_element(source, object_fit, Some(style)),
        }
    }
}

/// An overlay filling its container: the free placement.
fn fill(overlay: AnyElement) -> AnyElement {
    div()
        .absolute()
        .size_full()
        .child(overlay)
        .into_any_element()
}

/// An overlay pinned across the bottom of its container: where an HTML5 bar sits. The bar's own
/// (semitransparent) background is what keeps its text legible over the picture.
fn pin(overlay: AnyElement) -> AnyElement {
    div()
        .absolute()
        .bottom_0()
        .left_0()
        .w_full()
        .child(overlay)
        .into_any_element()
}

/// The frame itself: the surface when there is one, else a box of the same style — nothing ready yet,
/// or a stream between frames — so the view keeps its place in the layout.
fn frame_element(
    source: Option<SurfaceSource>,
    object_fit: ObjectFit,
    style: Option<StyleRefinement>,
) -> AnyElement {
    match source {
        Some(source) => sized(surface(source).object_fit(object_fit), style),
        None => sized(div(), style),
    }
}

/// Apply `style` to `element`, or stretch it to fill its container when there is none.
fn sized<E: Styled + IntoElement>(element: E, style: Option<StyleRefinement>) -> AnyElement {
    match style {
        Some(style) => {
            let mut element = element;
            *element.style() = style;
            element.into_any_element()
        }
        None => element.size_full().into_any_element(),
    }
}

/// The frame of a view-owned source: the player the facade opened, kept in an entity of its own so
/// both the picture and the controls can read it. Returns the frame to show and, when the default
/// controls were asked for, the bar to draw over it.
fn owned_player(
    window: &mut Window,
    cx: &mut App,
    id: ElementId,
    source: Source,
    wants_default_controls: bool,
) -> (Option<SurfaceSource>, Option<AnyElement>) {
    window.with_global_id(id, |global_id, window| {
        window.with_element_state::<Entity<OwnedPlayer>, _>(global_id, |state, window| {
            let player = match state {
                Some(player) => player,
                None => cx.new(|cx| OwnedPlayer::new(source, cx)),
            };

            // The view is where the visibility the pacing needs comes from: a frame taken for a
            // window that cannot show it is never released, and a bounded playback waits on it.
            let visible = window.is_visible();
            player.update(cx, |player, _| player.tick(visible));
            if player.read(cx).is_playing() {
                window.request_animation_frame();
            }

            let surface = player.update(cx, |player, _| player.surface());
            let bar = wants_default_controls
                .then(|| video_controls(player.clone()).into_any_element());
            ((surface, bar), player)
        })
    })
}

/// The frame of a view-owned producer, opening it on the first frame and keeping it in element state
/// under `id` thereafter.
fn owned_source(
    window: &mut Window,
    id: ElementId,
    open: Box<dyn FnOnce() -> Box<dyn SurfaceProducer>>,
) -> Option<SurfaceSource> {
    let mut open = Some(open);
    window.with_global_id(id, |global_id, window| {
        window.with_element_state::<Box<dyn SurfaceProducer>, _>(global_id, |state, _window| {
            let mut producer = match state {
                Some(producer) => producer,
                None => (open.take().expect("opened only when there is no state"))(),
            };
            let source = producer.surface();
            (source, producer)
        })
    })
}
