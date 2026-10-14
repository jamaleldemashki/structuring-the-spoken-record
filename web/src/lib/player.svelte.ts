// One audio element for the whole app, controlled straight from click handlers.
//
// Browsers only start media inside a user gesture, and with preload="none" nothing is
// fetched until play() is called. So playFrom() calls play() right away in the click and
// seeks once the duration is known (or at once if it already is). My first version waited
// for "loadedmetadata" before play(), which never fires for an episode not loaded yet.

export interface Clip {
  url: string;
  start: number;
}

export class Player {
  now = $state(0);
  duration = $state(0);
  paused = $state(true);
  failed = $state(false);
  /** The passage the user last asked to hear. */
  passageId = $state<string | null>(null);

  #audio: HTMLAudioElement | null = null;
  #url = "";
  #pendingSeek: number | null = null;

  #element(): HTMLAudioElement {
    if (this.#audio) return this.#audio;
    const a = new Audio();
    a.preload = "none";
    a.addEventListener("timeupdate", () => (this.now = a.currentTime));
    a.addEventListener("durationchange", () => (this.duration = Number.isFinite(a.duration) ? a.duration : 0));
    a.addEventListener("play", () => (this.paused = false));
    a.addEventListener("pause", () => (this.paused = true));
    a.addEventListener("loadedmetadata", () => this.#applySeek());
    a.addEventListener("error", () => { if (this.#url) this.failed = true; });
    return (this.#audio = a);
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

  seek(seconds: number) {
    if (this.#audio && this.duration) this.#audio.currentTime = Math.max(0, Math.min(this.duration, seconds));
  }
}

export const player = new Player();
