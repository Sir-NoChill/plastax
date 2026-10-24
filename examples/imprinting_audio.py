"""The audio prediction benchmark (thesis section 9.3) from synthesized audio.

Synthesizes a one-hour-like audio stream at 16,384 Hz -- a coloured-noise
background with three note-like sounds (piano-ish tones with harmonics and
an exponential decay, by default C4, E4 and D4 as in the thesis's C4/E4
problem), each followed after 3-5 s by a reward of +1, -1 or nothing -- and
runs the thesis's preprocessing: every step advances 640 samples (about
40 ms), an FFT over the last 1,024 samples gives 512 magnitudes, drawn on a
50 x 50 grid of frequency x magnitude; a cell is 1 where the curve passes
through it, so every frame has exactly 50 ones among 2,500 bits. The output
is a `Stream` in the same files the learners read.

One conceptual point: the stream's statistics (a stable but not constant
background, sounds that occupy their own grid cells for about 25 frames,
rewards long after the sound) are what the thesis's hyperparameters were
tuned on; the synthetic bit stream of `imprinting_stream.py` is harsher in
ways that matter under the overshoot bound.

Run:  uv run python examples/imprinting_audio.py --minutes 60 --out audio60
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import imprinting_stream as ims
import numpy as np

RATE = 16_384
HOP = 640
WINDOW = 1024
GRID = 50
N_OBS = GRID * GRID


@dataclasses.dataclass(frozen=True)
class AudioConfig:
    """The synthesized benchmark's parameters.

    Attributes:
        minutes: stream length in minutes of audio (the thesis used one hour).
        notes_hz: the three sounds' fundamentals, rewarded (+1, -1, 0) in order.
        note_seconds: a note's length.
        delay_seconds: inclusive range of the sound-end-to-reward delay.
        gap_seconds: inclusive range of the silence between sounds.
        noise_level: RMS of the background noise relative to a note's peak.
        hum_hz: a steady background tone (0 for none), as real rooms have.
        magnitude_scale: FFT magnitudes are divided by this before being
            mapped to the 50 grid rows; rows above the top are clipped.
        seed: generator seed.
    """

    minutes: float = 60.0
    notes_hz: tuple[float, float, float] = (261.63, 329.63, 293.66)
    note_seconds: float = 1.0
    delay_seconds: tuple[float, float] = (3.0, 5.0)
    gap_seconds: tuple[float, float] = (15.0, 30.0)
    noise_level: float = 0.05
    hum_hz: float = 60.0
    magnitude_scale: float = 0.15
    seed: int = 0


def _note(rng: np.random.Generator, hz: float, seconds: float) -> np.ndarray:
    """A piano-like tone: harmonics with decaying amplitudes, exponential decay."""
    t = np.arange(int(seconds * RATE)) / RATE
    out = np.zeros_like(t)
    for k in range(1, 7):
        amp = 1.0 / k**1.5
        detune = 1.0 + 0.0005 * rng.standard_normal()
        out += amp * np.sin(2 * np.pi * hz * k * detune * t)
    env = np.exp(-3.0 * t) * np.minimum(1.0, t * 200.0)
    return out * env / np.max(np.abs(out) + 1e-9)


def _coloured_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Pink-ish noise: white noise shaped by 1/sqrt(f) in the spectrum."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.shape[0])
    spec[1:] /= np.sqrt(f[1:])
    out = np.fft.irfft(spec, n)
    return out / (np.std(out) + 1e-9)


def synthesize(cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The audio, the per-step rewards, the events and the step count.

    Args:
        cfg: the parameters.

    Returns:
        `(audio, reward, events, steps)`: float32 samples, (steps,) rewards,
        (n_events, 4) int32 rows (sound, t_on, t_off, t_reward) in steps.
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.minutes * 60 * RATE)
    steps = (n - WINDOW) // HOP + 1
    audio = cfg.noise_level * _coloured_noise(rng, n)
    if cfg.hum_hz > 0:
        audio += 0.02 * np.sin(2 * np.pi * cfg.hum_hz * np.arange(n) / RATE)
    reward = np.zeros((steps,), np.float32)
    rewards = (1.0, -1.0, 0.0)
    events: list[tuple[int, int, int, int]] = []
    pos = int(rng.uniform(*cfg.gap_seconds) * RATE)
    while True:
        sound = int(rng.integers(3))
        note = _note(rng, cfg.notes_hz[sound], cfg.note_seconds)
        end = pos + note.shape[0]
        r_pos = end + int(rng.uniform(*cfg.delay_seconds) * RATE)
        if r_pos >= n - WINDOW:
            break
        audio[pos:end] += note
        t_on = max(0, (pos - WINDOW) // HOP + 1)
        t_off = (end - WINDOW) // HOP + 1
        t_r = (r_pos - WINDOW) // HOP + 1
        reward[t_r] = rewards[sound]
        events.append((sound, t_on, t_off, t_r))
        pos = end + int(rng.uniform(*cfg.gap_seconds) * RATE)
    return (
        audio.astype(np.float32),
        reward,
        np.asarray(events, np.int32).reshape(-1, 4),
        steps,
    )


def grid_frames(audio: np.ndarray, steps: int, magnitude_scale: float) -> np.ndarray:
    """The thesis's preprocessing: FFT of the last 1,024 samples on a 50 x 50 grid.

    Frequency bins 1..512 fall into 50 columns of about 10 bins; a column's
    magnitude is the mean over its bins, divided by `magnitude_scale`, and the
    row it lands on (clipped to the top row) is the one cell set, so every
    frame has exactly 50 ones.

    Args:
        audio: the samples.
        steps: frames to produce (frame `t` ends at sample `t * HOP + WINDOW`).
        magnitude_scale: divisor mapping magnitudes onto the 50 rows.

    Returns:
        (steps, 2500) uint8 frames, row-major over (column, row).
    """
    window = np.hanning(WINDOW).astype(np.float32)
    edges = np.linspace(1, WINDOW // 2 + 1, GRID + 1).astype(int)
    out = np.zeros((steps, N_OBS), np.uint8)
    for t in range(steps):
        end = t * HOP + WINDOW
        seg = audio[end - WINDOW : end] * window
        mag = np.abs(np.fft.rfft(seg))[1:]  # bins 1..512
        cols = np.asarray(
            [mag[edges[c] - 1 : edges[c + 1] - 1].mean() for c in range(GRID)]
        )
        rows = np.clip((cols / magnitude_scale).astype(int), 0, GRID - 1)
        out[t, np.arange(GRID) * GRID + rows] = 1
    return out


def make_stream(cfg: AudioConfig, gamma: float = 0.99) -> ims.Stream:
    """Synthesize, preprocess and package as a `Stream`."""
    audio, reward, events, steps = synthesize(cfg)
    obs = grid_frames(audio, steps, cfg.magnitude_scale)
    return ims.Stream(obs=obs, reward=reward, events=events, gamma=gamma)


def main(argv: list[str] | None = None) -> None:
    """Write the synthesized benchmark as .npz and .bin, and describe it."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--minutes", type=float, default=AudioConfig.minutes)
    p.add_argument("--seed", type=int, default=AudioConfig.seed)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--noise-level", type=float, default=AudioConfig.noise_level)
    p.add_argument("--magnitude-scale", type=float, default=AudioConfig.magnitude_scale)
    a = p.parse_args(argv)
    cfg = AudioConfig(
        minutes=a.minutes,
        seed=a.seed,
        noise_level=a.noise_level,
        magnitude_scale=a.magnitude_scale,
    )
    stream = make_stream(cfg, a.gamma)
    on_rows = stream.obs.sum(axis=1)
    assert np.all(on_rows == GRID)
    # How distinct are the sounds from the background and from each other?
    silence = stream.obs[: int(stream.events[0, ims.EVENT_ON])].mean(axis=0)
    print(
        f"audio benchmark: {stream.obs.shape[0]} steps, "
        f"{stream.events.shape[0]} events, "
        f"{N_OBS} bits with {GRID} on per frame"
    )
    for sound in range(3):
        evs = [e for e in stream.events if e[0] == sound]
        frames = np.concatenate([stream.obs[e[1] : e[2]] for e in evs[:20]])
        mean = frames.mean(axis=0)
        own = np.flatnonzero((mean > 0.3) & (silence < 0.05))
        print(
            f"  sound {sound}: {len(evs)} events, {evs[0][2] - evs[0][1]} frames each, "
            f"{own.size} cells it owns against the background"
        )
    ims.save_npz(stream, a.out.with_suffix(".npz"))
    ims.save_binary(stream, a.out.with_suffix(".bin"))
    print(f"wrote {a.out.with_suffix('.npz')} and {a.out.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
