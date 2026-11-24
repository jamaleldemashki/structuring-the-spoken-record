// One audio element for the whole app, controlled straight from click handlers.
//
// Browsers only start media inside a user gesture, and with preload="none" nothing is
// fetched until play() is called. So playFrom() calls play() right away in the click and
// seeks once the duration is known (or at once if it already is). My first version waited
// for "loadedmetadata" before play(), which never fires for an episode not loaded yet.

export interface Clip {
  url: string;
  start: number;
  /** Where the passage ends. Playing stops there. */
  end?: number | null;
}

export class Player {
  now = $state(0);
  duration = $state(0);
  paused = $state(true);
  failed = $state(false);
  /** The passage the user last asked to hear. */
  passageId = $state<string | null>(null);
  /** Playing stops at the end of the passage. */
  stopsAtEnd = $state(false);
  rate = $state(1);

  #audio: HTMLAudioElement | null = null;
  #url = "";
  #pendingSeek: number | null = null;
  // The passage being played, so playing can stop at its end. Cleared when the listener
  // seeks outside it or presses play again after the stop.
  #clip: { start: number; end: number } | null = null;

  #element(): HTMLAudioElement {
    if (this.#audio) return this.#audio;
    const a = new Audio();
    a.preload = "none";
    a.addEventListener("timeupdate", () => { this.now = a.currentTime; this.#stopAtEnd(); });
    a.addEventListener("durationchange", () => (this.duration = Number.isFinite(a.duration) ? a.duration : 0));
    a.addEventListener("play", () => { this.paused = false; this.#tick(); });
    a.addEventListener("pause", () => (this.paused = true));
    a.addEventListener("loadedmetadata", () => this.#applySeek());
    a.addEventListener("error", () => { if (this.#url) this.failed = true; });
    return (this.#audio = a);
  }

  // "timeupdate" only fires about four times a second, too slow to follow single words.
  // While playing, the time is read once per frame instead.
  #tick() {
    const a = this.#audio;
    if (!a || a.paused) return;
    this.now = a.currentTime;
    this.#stopAtEnd();
    requestAnimationFrame(() => this.#tick());
  }

  #stopAtEnd() {
    const a = this.#audio;
    if (!a || !this.#clip || a.paused || a.currentTime < this.#clip.end) return;
    a.pause();
    // The browser reports the pause a moment later. Until then this must not look like playing on.
    this.paused = true;
    this.keepPlaying();
  }

  /** Plays on past the end of the passage, like a normal podcast player. */
  keepPlaying() {
    this.#clip = null;
    this.stopsAtEnd = false;
  }

  setRate(rate: number) {
    this.rate = rate;
    if (this.#audio) this.#audio.playbackRate = rate;
  }

  #applySeek() {
    if (this.#pendingSeek === null || !this.#audio) return;
    this.#audio.currentTime = this.#pendingSeek;
    this.now = this.#pendingSeek;
    this.#pendingSeek = null;
  }

  /** Start playing at a passage. Call it from the click handler itself. */
  playFrom(passageId: string, clip: Clip) {
    const a = this.#element();
    this.passageId = passageId;
    this.failed = false;
    if (clip.url !== this.#url) {
      this.#url = clip.url;
      this.duration = 0;
      a.src = clip.url;
    }
    this.#pendingSeek = clip.start;
    // The time of whatever played before no longer says anything.
    this.now = clip.start;
    this.#clip = clip.end != null && clip.end > clip.start ? { start: clip.start, end: clip.end } : null;
    this.stopsAtEnd = !!this.#clip;
    a.playbackRate = this.rate;
    // readyState HAVE_METADATA or later, so we can seek right away.
    if (a.readyState >= 1) this.#applySeek();
    a.play().catch((err: DOMException) => {
      // A newer playFrom() with another source aborts this request. That's fine.
      if (err.name !== "AbortError") this.failed = true;
    });
  }

  toggle() {
    const a = this.#audio;
    if (!a || !this.#url) return;
    if (a.paused) a.play().catch(() => (this.failed = true));
    else a.pause();
  }

  pause() {
    this.#audio?.pause();
  }

  seek(seconds: number) {
    if (this.#clip && (seconds < this.#clip.start || seconds >= this.#clip.end)) this.keepPlaying();
    if (this.#audio && this.duration) this.#audio.currentTime = Math.max(0, Math.min(this.duration, seconds));
  }
}

export const player = new Player();
