//! The transport bar: an HTML5-shaped control surface over a [`Player`].
//!
//! This is the *default* rendering of the transport, and nothing more than that: the headless state
//! is [`Transport`], and a user-defined bar reads the same snapshot and draws its own pixels instead.
//! Pass either to [`Video::controls`](crate::Video::controls) to place it over the frame.
//!
//! [`video_controls(player)`](video_controls) reads its whole state from the same entity
//! [`video(player)`](crate::video) draws and writes back through the [`Player`] transport: play/pause
//! toggles, and the scrub rail seeks — by click or by drag — when the player says it can. It is
//! deliberately *stateless* — no scrub latch of its own, because a drag reports the pressed button in
//! the move event — so it can sit beside the picture without claiming a second identity on the same
//! entity; the rail is inert when [`Player::is_seekable`] is false.

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use gpui_authoring::{
    AnyElement, App, Bounds, Corners, ElementId, Entity, EntityId, InteractiveElement, IntoElement,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, Refineable, Render,
    StatefulInteractiveElement, StyleRefinement, Styled, View, ViewElement, Window, canvas, div,
    fill, point, px, rgb, rgba, size,
};

use crate::Player;

/// What [`Video::controls`](crate::Video::controls) draws over the frame.
///
/// The `From` impls are what let one method take `true`, `false`, [`video_controls(player)`] (the
/// default bar), or a user-defined element — an [`Entity`] an app built for its own controls, or a
/// bare [`AnyElement`]. A user-defined element that is neither goes through
/// [`into_any_element`](IntoElement::into_any_element); that last case is how a custom bar replaces
/// the default by being passed in its place.
pub enum Controls {
    /// Nothing at all.
    None,
    /// The default HTML5 bar, [`video_controls`] — when there is a player entity to read.
    Default,
    /// A caller-supplied element, drawn in the bar's place, pinned across the bottom edge.
    Custom(AnyElement),
}

impl From<bool> for Controls {
    fn from(show: bool) -> Self {
        if show {
            Controls::Default
        } else {
            Controls::None
        }
    }
}

impl From<AnyElement> for Controls {
    fn from(controls: AnyElement) -> Self {
        Controls::Custom(controls)
    }
}

impl<P: Player> From<VideoControls<P>> for Controls {
    fn from(controls: VideoControls<P>) -> Self {
        Controls::Custom(controls.into_any_element())
    }
}

impl<V: Render> From<Entity<V>> for Controls {
    fn from(controls: Entity<V>) -> Self {
        Controls::Custom(controls.into_any_element())
    }
}

/// A snapshot of a player's transport — the headless half of the controls.
///
/// [`video_controls`] reads one of these and draws it; a user-defined bar reads the same thing and
/// draws whatever it likes, so the two differ only in paint. Take one with [`Transport::read`] and
/// call the [`Player`] transport methods to drive it back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transport {
    /// Whether playback is running.
    pub playing: bool,
    /// Where playback is.
    pub position: Duration,
    /// How long the stream is, or `None` when it has no length to report.
    pub duration: Option<Duration>,
    /// Whether the player will honour a seek.
    pub seekable: bool,
    /// `position / duration`, clamped to `0.0..=1.0`; `0.0` when there is no duration.
    pub progress: f32,
}

impl Transport {
    /// Read the transport of `player`.
    pub fn read<P: Player>(player: &Entity<P>, cx: &App) -> Self {
        let player = player.read(cx);
        let position = player.position();
        let duration = player.duration();
        let progress = duration
            .filter(|duration| !duration.is_zero())
            .map(|duration| (position.as_secs_f32() / duration.as_secs_f32()).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        Self {
            playing: player.is_playing(),
            position,
            duration,
            seekable: player.is_seekable(),
            progress,
        }
    }
}

/// The controls' transport bar. Built by [`video_controls`].
pub struct VideoControls<P: Player> {
    player: Entity<P>,
    style: StyleRefinement,
}

/// A transport bar for `player`: play/pause, a scrub rail, and a timecode.
///
/// It re-renders whenever the entity it reads is notified, because reading it during render is what
/// marks the enclosing view as an accessor. Pass it to [`Video::controls`](crate::Video::controls)
/// (or [`Video::overlay`](crate::Video::overlay) for free placement), or build your own from
/// [`Transport`] and pass that instead.
pub fn video_controls<P: Player>(player: Entity<P>) -> VideoControls<P> {
    VideoControls {
        player,
        style: StyleRefinement::default(),
    }
}

impl<P: Player> Styled for VideoControls<P> {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl<P: Player> IntoElement for VideoControls<P> {
    type Element = ViewElement<VideoControls<P>>;

    fn into_element(self) -> Self::Element {
        ViewElement::new(self)
    }
}

impl<P: Player> View for VideoControls<P> {
    fn entity_id(&self) -> Option<EntityId> {
        // Stateless on purpose: the picture already holds the entity's identity, and two views keyed
        // on one entity must not be siblings. Controls read the player without owning its id.
        None
    }

    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let VideoControls { player, style } = self;
        let id = player.entity_id().as_u64();
        let transport = Transport::read(&player, cx);

        let toggle = player.clone();
        let timecode = format!(
            "{} / {}",
            timecode(transport.position),
            transport
                .duration
                .map(timecode)
                .unwrap_or_else(|| "--:--".to_owned()),
        );

        let mut bar = div()
            .flex()
            .items_center()
            .gap_3()
            .px_3()
            .py_2()
            // Semitransparent, so the picture reads through the bar while it sits over it — the
            // scrim an HTML5 bar has over a frame, rather than an opaque strip.
            .bg(rgba(0x101014a6))
            .text_color(rgb(0xd8d8e0))
            .text_sm()
            .child(
                div()
                    .id(ElementId::NamedInteger("video-controls-play".into(), id))
                    .w(px(34.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_md()
                    .bg(rgb(0x2e2e38))
                    .hover(|style| style.bg(rgb(0x44444f)))
                    .cursor_pointer()
                    .child(if transport.playing { "Pause" } else { "Play" })
                    .on_click(move |_, _window, cx| {
                        if std::env::var_os("GPUI_VIDEO_TRACE").is_some() {
                            eprintln!("gpui_video: controls: play/pause button clicked");
                        }
                        toggle.update(cx, |player, cx| {
                            player.toggle();
                            cx.notify();
                        });
                    }),
            );

        // The rail measures progress against a length. A source that declares none — a raw
        // elementary stream, which is unbounded until it stops — gets a spacer instead, so the bar
        // stays laid out and no one is told a position the stream cannot give.
        if transport.duration.is_some() {
            bar = bar.child(rail(player, id, transport.progress, transport.seekable));
        } else {
            bar = bar.child(div().flex_1());
        }
        bar = bar.child(div().min_w(px(104.0)).text_right().child(timecode));

        bar.style().refine(&style);
        bar
    }
}

/// The scrub rail: a canvas that draws the track, the elapsed fill and the thumb, over a grab area
/// that turns a press, a drag, or a release into a seek.
///
/// A press starts the scrub and pauses; while it is held, moves are tracked **window-wide**, not just
/// over the rail — a drag that wanders far up off the thin bar keeps scrubbing — and the release ends
/// it wherever it lands. Both come from [`Window::on_mouse_event`] listeners registered while
/// painting, which is window-wide by construction and needs no pointer capture; the rail's x is read
/// from the bounds the canvas recorded.
fn rail<P: Player>(player: Entity<P>, id: u64, progress: f32, seekable: bool) -> impl IntoElement {
    let bounds: Rc<Cell<(f32, f32)>> = Rc::new(Cell::new((0.0, 1.0)));
    let recorded = bounds.clone();
    let grabbed = bounds.clone();
    let tracked = bounds;

    let grab_player = player.clone();
    let paint_player = player.clone();

    let mut rail = div()
        .id(ElementId::NamedInteger("video-controls-rail".into(), id))
        .relative()
        .flex_1()
        .h(px(20.0));

    if seekable {
        rail = rail.cursor_pointer().on_mouse_down(
            MouseButton::Left,
            move |event: &MouseDownEvent, _window, cx| {
                if std::env::var_os("GPUI_VIDEO_TRACE").is_some() {
                    eprintln!("gpui_video: controls: rail grabbed");
                }
                let x = f32::from(event.position.x);
                let (origin, width) = grabbed.get();
                let fraction = ((x - origin) / width.max(1.0)).clamp(0.0, 1.0);
                grab_player.update(cx, |player, cx| {
                    player.scrub_started();
                    if let Some(duration) = player.duration() {
                        player.seek(duration.mul_f32(fraction));
                    }
                    cx.notify();
                });
            },
        );
    }

    rail.child(
        canvas(
            |_, _, _| {},
            move |bounds, _, window, _cx| {
                recorded.set((f32::from(bounds.origin.x), f32::from(bounds.size.width)));

                let center_y = bounds.origin.y + bounds.size.height / 2.0;
                let bar_h = 4.0;
                let bar_y = center_y - px(bar_h / 2.0);
                let radius = Corners::all(px(bar_h / 2.0));

                window.paint_quad(
                    fill(
                        Bounds {
                            origin: point(bounds.origin.x, bar_y),
                            size: size(bounds.size.width, px(bar_h)),
                        },
                        rgb(0x2c2c34),
                    )
                    .corner_radii(radius),
                );

                let filled = f32::from(bounds.size.width) * progress.clamp(0.0, 1.0);
                if filled > 0.0 {
                    window.paint_quad(
                        fill(
                            Bounds {
                                origin: point(bounds.origin.x, bar_y),
                                size: size(px(filled), px(bar_h)),
                            },
                            rgb(0x4a9eff),
                        )
                        .corner_radii(radius),
                    );
                }

                let thumb_r = 5.0;
                let thumb_x = bounds.origin.x + px(filled);
                window.paint_quad(
                    fill(
                        Bounds {
                            origin: point(thumb_x - px(thumb_r), center_y - px(thumb_r)),
                            size: size(px(thumb_r * 2.0), px(thumb_r * 2.0)),
                        },
                        rgb(0xf0f0f5),
                    )
                    .corner_radii(Corners::all(px(thumb_r))),
                );

                if !seekable {
                    return;
                }

                let dragging = paint_player.clone();
                let dragging_bounds = tracked.clone();
                window.on_mouse_event(move |event: &MouseMoveEvent, _phase, _window, cx| {
                    if event.pressed_button == Some(MouseButton::Left)
                        && dragging.read(cx).scrubbing()
                    {
                        seek(&dragging, &dragging_bounds, f32::from(event.position.x), cx);
                    }
                });

                let released = paint_player;
                window.on_mouse_event(move |event: &MouseUpEvent, _phase, _window, cx| {
                    if event.button == MouseButton::Left {
                        released.update(cx, |player, cx| {
                            if player.scrubbing() {
                                if std::env::var_os("GPUI_VIDEO_TRACE").is_some() {
                                    eprintln!("gpui_video: controls: rail released");
                                }
                                player.scrub_ended();
                                cx.notify();
                            }
                        });
                    }
                });
            },
        )
        .size_full(),
    )
}

/// Seek `player` to the fraction of its length that `x` falls at, using the rail bounds recorded
/// while painting.
fn seek<P: Player>(player: &Entity<P>, bounds: &Rc<Cell<(f32, f32)>>, x: f32, cx: &mut App) {
    let (origin, width) = bounds.get();
    let fraction = ((x - origin) / width.max(1.0)).clamp(0.0, 1.0);
    player.update(cx, |player, cx| {
        if let Some(duration) = player.duration() {
            player.seek(duration.mul_f32(fraction));
            cx.notify();
        }
    });
}

/// `M:SS`, for a timecode that means something at a glance.
fn timecode(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}
