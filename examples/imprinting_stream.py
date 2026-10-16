"""Synthetic audio-like prediction stream for the imprinting learner.

A scaled-down analogue of the audio prediction benchmark (Javed 2025, thesis
section 9.3): a stream of binary observation frames in which three "sounds"
-- fixed short sequences of frames, replayed with jitter -- are each followed
after a delay by a scalar reward (+1, -1 or none). The reward arrives long
after the sound has ended, so predicting it requires features that remember
the sound; a learner that only sees the current frame can predict the return
while the sound is audible and not afterwards (thesis Figure 9.4 vs 9.5).

One conceptual point: the stream is the *problem*, shared byte-for-byte by
the plastax and plastax-cpp drivers, and the GVF-shape metrics below are the
*acceptance criterion* for a learner on it: the prediction has to leave zero
as soon as the sound plays, track the discounted return up to the reward,
and drop back to zero right after it.

Run:  uv run python examples/imprinting_stream.py --out stream --steps 60000
writes `stream.npz` (plastax) and `stream.bin` (plastax-cpp) and prints the
event table. No learner is involved; see `imprinting_learner.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import struct
from pathlib import Path

import numpy as np

# The plastax-cpp reader (examples/imprinting-learner/stream.hpp) parses this
# header; keep the two in sync.
_BIN_MAGIC = b"IMPR"
_BIN_VERSION = 1


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    """The stream's parameters (thesis analogues in the plan, section 4.1).

    Attributes:
        n_obs: observation bits per frame.
        active_bits: bits set in every frame (sound and silence alike).
        rewards: the reward each sound is followed by; one sound per entry.
        sound_len: frames per sound.
        jitter: probability that each active bit of a sound frame moves to a
            random free position in a given replay.
        background_jitter: the same for the silence frames, which replay one
            fixed background template (a steady noise spectrum) rather than
            fresh random bits, so their step sizes settle as real
            background does.
        background_drift: per silence step, the probability that each
            template bit moves for good to a random free position, so the
            background spectrum drifts slowly instead of standing still.
        gap: inclusive range, in steps, of the silence between the end of a
            sound and the start of the next one.
        delay: inclusive range, in steps, from the end of a sound to its
            reward step.
        steps: stream length.
        gamma: the discount of the prediction problem (a property of the
            problem, as in the thesis).
        seed: generator seed.
    """

    n_obs: int = 400
    active_bits: int = 8
    rewards: tuple[float, ...] = (1.0, -1.0, 0.0)
    sound_len: int = 12
    jitter: float = 0.1
    background_jitter: float = 0.3
    background_drift: float = 0.0
    gap: tuple[int, int] = (150, 300)
    delay: tuple[int, int] = (40, 80)
    steps: int = 60_000
    gamma: float = 0.99
    seed: int = 0


# One event = one sound replay: (sound index, first frame, one past the last
# frame, reward step). Rows of a (n_events, 4) int32 array.
EVENT_SOUND, EVENT_ON, EVENT_OFF, EVENT_REWARD = range(4)


@dataclasses.dataclass(frozen=True)
class Stream:
    """A generated stream.

    Attributes:
        obs: (steps, n_obs) uint8 frames.
        reward: (steps,) float32 rewards; nonzero on reward steps only.
        events: (n_events, 4) int32 rows (sound, t_on, t_off, t_reward).
        gamma: the problem's discount.
    """

    obs: np.ndarray
    reward: np.ndarray
    events: np.ndarray
    gamma: float


def _frame(
    rng: np.random.Generator, n_obs: int, active: np.ndarray, jitter: float
) -> np.ndarray:
    """One frame from a set of active bit positions, with per-bit jitter."""
    bits = set(int(b) for b in active)
    if jitter > 0.0:
        moved = rng.random(active.shape[0]) < jitter
        for b in active[moved]:
            bits.discard(int(b))
            # A free position so the frame keeps exactly `active_bits` ones.
            while True:
                candidate = int(rng.integers(n_obs))
                if candidate not in bits:
                    bits.add(candidate)
                    break
    frame = np.zeros((n_obs,), np.uint8)
    frame[list(bits)] = 1
    return frame


def generate(cfg: StreamConfig) -> Stream:
    """Generate a stream deterministically from its config.

    Args:
        cfg: the stream parameters.

    Returns:
        The stream; events whose reward step would fall past `cfg.steps` are
        not started, so every listed event completes inside the stream.
    """
    rng = np.random.default_rng(cfg.seed)
    n_sounds = len(cfg.rewards)
    templates = [
        [
            rng.choice(cfg.n_obs, cfg.active_bits, replace=False)
            for _ in range(cfg.sound_len)
        ]
        for _ in range(n_sounds)
    ]
    obs = np.zeros((cfg.steps, cfg.n_obs), np.uint8)
    reward = np.zeros((cfg.steps,), np.float32)
    events: list[tuple[int, int, int, int]] = []
    t = 0
    while True:
        t_on = t + int(rng.integers(cfg.gap[0], cfg.gap[1] + 1))
        sound = int(rng.integers(n_sounds))
        t_off = t_on + cfg.sound_len
        t_reward = t_off + int(rng.integers(cfg.delay[0], cfg.delay[1] + 1))
        if t_reward >= cfg.steps:
            break
        for k in range(cfg.sound_len):
            obs[t_on + k] = _frame(rng, cfg.n_obs, templates[sound][k], cfg.jitter)
        reward[t_reward] = np.float32(cfg.rewards[sound])
        events.append((sound, t_on, t_off, t_reward))
        t = t_off
    # Silence: every frame not covered by a sound replays the background
    # template with its own jitter.
    background = rng.choice(cfg.n_obs, cfg.active_bits, replace=False)
    covered = np.zeros((cfg.steps,), bool)
    for _, t_on, t_off, _ in events:
        covered[t_on:t_off] = True
    for s in np.flatnonzero(~covered):
        if cfg.background_drift > 0.0:
            moved = np.flatnonzero(
                rng.random(background.shape[0]) < cfg.background_drift
            )
            for k in moved:
                while True:
                    candidate = int(rng.integers(cfg.n_obs))
                    if candidate not in background:
                        background[k] = candidate
                        break
        obs[s] = _frame(rng, cfg.n_obs, background, cfg.background_jitter)
    return Stream(
        obs=obs,
        reward=reward,
        events=np.asarray(events, np.int32).reshape(-1, 4),
        gamma=cfg.gamma,
    )


def discounted_return(reward: np.ndarray, gamma: float) -> np.ndarray:
    """The return `G_t = r_{t+1} + gamma G_{t+1}` the prediction at `t` targets.

    The thesis's lifetime error (SwiftTD paper, eq. 3) compares `v_t` with
    `sum_{j > t} gamma^(j-t-1) r_j`: the reward on step `t_r` pays the
    prediction made on `t_r - 1`, and the return is 0 from `t_r` on until the
    next reward. Truncated at the end of the stream.

    Args:
        reward: (steps,) rewards.
        gamma: discount.

    Returns:
        (steps,) float32 returns.
    """
    out = np.zeros(reward.shape, np.float32)
    acc = np.float32(0.0)
    for t in range(reward.shape[0] - 2, -1, -1):
        acc = np.float32(reward[t + 1] + gamma * acc)
        out[t] = acc
    return out


def learnable_return(stream: Stream) -> np.ndarray:
    """The best learnable prediction: the return from a sound's onset to its reward.

    The true return is nonzero long before a sound (the reward is coming
    whether or not the cue has played), but nothing in the stream predicts it
    until the sound starts (thesis Figure 9.3, "best learnable prediction").
    So the target the GVF-shape criterion and the plots use is `G_t` on
    `[t_on, t_reward)` of every rewarded event and 0 elsewhere.

    Args:
        stream: the stream.

    Returns:
        (steps,) float32 target.
    """
    ret = discounted_return(stream.reward, stream.gamma)
    out = np.zeros_like(ret)
    for _, t_on, _, t_r in stream.events:
        if stream.reward[t_r] != 0.0:
            out[t_on:t_r] = ret[t_on:t_r]
    return out


def lifetime_error(prediction: np.ndarray, ret: np.ndarray) -> float:
    """Mean squared error between the predictions and the returns."""
    diff = np.asarray(prediction, np.float64) - np.asarray(ret, np.float64)
    return float(np.mean(diff * diff))


# --- GVF-shape metrics (plan section 6) -----------------------------------


@dataclasses.dataclass(frozen=True)
class ShapeThresholds:
    """Thresholds of the GVF-shape criterion, as fractions of |reward|.

    Attributes:
        quiet: |v| below this counts as "zero" (before onset, after drop,
            and on the unrewarded sound).
        pre_window: steps before onset that must be quiet.
        onset_fraction: |v| at the end of the sound must reach this fraction
            of |G| there.
        window_rmse: RMSE of v - G over [t_off, t_reward) must be below this.
        sign_fraction: fraction of that window on which sign(v) = sign(r).
        drop_steps: v must be quiet from this many steps after the reward.
    """

    quiet: float = 0.1
    pre_window: int = 20
    onset_fraction: float = 0.25
    window_rmse: float = 0.15
    sign_fraction: float = 0.9
    drop_steps: int = 3


# Columns of the per-event table returned by `shape_metrics`.
SHAPE_COLUMNS = (
    "event",
    "sound",
    "reward",
    "t_on",
    "t_reward",
    "pre_quiet",
    "onset",
    "window_rmse",
    "sign_frac",
    "drop",
    "pass",
)


def shape_metrics(
    prediction: np.ndarray,
    stream: Stream,
    *,
    converged_fraction: float = 0.25,
    thresholds: ShapeThresholds | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Score every late event of the stream against the GVF-shape criterion.

    Args:
        prediction: (steps,) the learner's prediction `v_t`.
        stream: the stream it was made on.
        converged_fraction: only events starting in the last this fraction
            of the stream are scored.
        thresholds: the criterion's thresholds.

    Returns:
        The per-event rows (keys `SHAPE_COLUMNS`; an unrewarded sound has
        only `pre_quiet`, `window_rmse` (over its would-be window) and
        `pass` = quiet throughout), and a summary: the fraction of rewarded
        events passing each sub-criterion and overall, the fraction of
        unrewarded events that stayed quiet, and the mean window RMSE.
    """
    v = np.asarray(prediction, np.float64)
    ret = learnable_return(stream).astype(np.float64)
    steps = v.shape[0]
    start = int(steps * (1.0 - converged_fraction))
    th = thresholds if thresholds is not None else ShapeThresholds()
    rows: list[dict[str, float]] = []
    events = stream.events
    for k in range(events.shape[0]):
        sound, t_on, t_off, t_r = (int(x) for x in events[k])
        if t_on < start:
            continue
        r = float(stream.reward[t_r])
        next_on = int(events[k + 1, EVENT_ON]) if k + 1 < events.shape[0] else steps
        pre = v[max(0, t_on - th.pre_window) : t_on]
        if r == 0.0:
            # Unrewarded: quiet before, during and over a nominal delay after.
            window = v[t_on : min(next_on, t_off + stream_delay_hint(stream))]
            quiet = (
                float(np.max(np.abs(pre))) < th.quiet
                and float(np.max(np.abs(window))) < th.quiet
            )
            rows.append(
                {
                    "event": k,
                    "sound": sound,
                    "reward": 0.0,
                    "t_on": t_on,
                    "t_reward": -1,
                    "pre_quiet": float(np.max(np.abs(pre)) < th.quiet),
                    "onset": float("nan"),
                    "window_rmse": float(np.sqrt(np.mean(window * window))),
                    "sign_frac": float("nan"),
                    "drop": float("nan"),
                    "pass": float(quiet),
                }
            )
            continue
        scale = abs(r)
        pre_quiet = float(np.max(np.abs(pre))) < th.quiet * scale if pre.size else True
        v_end = v[t_off - 1]
        onset = np.sign(v_end) == np.sign(r) and abs(v_end) >= th.onset_fraction * abs(
            ret[t_off - 1]
        )
        win_v = v[t_off:t_r]
        win_g = ret[t_off:t_r]
        rmse = float(np.sqrt(np.mean((win_v - win_g) ** 2)))
        sign_frac = float(np.mean(np.sign(win_v) == np.sign(r)))
        after = v[t_r + th.drop_steps + 1 : next_on]
        drop = float(np.max(np.abs(after))) < th.quiet * scale if after.size else True
        ok = (
            pre_quiet
            and bool(onset)
            and rmse <= th.window_rmse * scale
            and sign_frac >= th.sign_fraction
            and drop
        )
        rows.append(
            {
                "event": k,
                "sound": sound,
                "reward": r,
                "t_on": t_on,
                "t_reward": t_r,
                "pre_quiet": float(pre_quiet),
                "onset": float(onset),
                "window_rmse": rmse,
                "sign_frac": sign_frac,
                "drop": float(drop),
                "pass": float(ok),
            }
        )
    rewarded = [row for row in rows if row["reward"] != 0.0]
    unrewarded = [row for row in rows if row["reward"] == 0.0]

    def frac(key: str, table: list[dict[str, float]]) -> float:
        return float(np.mean([row[key] for row in table])) if table else float("nan")

    summary = {
        "n_rewarded": float(len(rewarded)),
        "n_unrewarded": float(len(unrewarded)),
        "pre_quiet": frac("pre_quiet", rewarded),
        "onset": frac("onset", rewarded),
        "window_rmse": frac("window_rmse", rewarded),
        "sign_frac": frac("sign_frac", rewarded),
        "drop": frac("drop", rewarded),
        "pass": frac("pass", rewarded),
        "unrewarded_quiet": frac("pass", unrewarded),
    }
    return rows, summary


def stream_delay_hint(stream: Stream) -> int:
    """The longest sound-end-to-reward delay among the stream's rewarded events."""
    ev = stream.events
    rewarded = ev[stream.reward[ev[:, EVENT_REWARD]] != 0.0]
    if rewarded.shape[0] == 0:
        return 0
    return int(np.max(rewarded[:, EVENT_REWARD] - rewarded[:, EVENT_OFF]))


def format_summary(summary: dict[str, float]) -> str:
    """One line per summary entry, for the console."""
    return "\n".join(f"  {k:>17s}: {val:.3f}" for k, val in summary.items())


# --- file formats ---------------------------------------------------------


def save_npz(stream: Stream, path: Path) -> None:
    """Write the stream as an .npz (the plastax driver's input)."""
    np.savez_compressed(
        path,
        obs=stream.obs,
        reward=stream.reward,
        events=stream.events,
        gamma=np.float32(stream.gamma),
    )


def load_npz(path: Path) -> Stream:
    """Read a stream written by `save_npz`."""
    with np.load(path) as data:
        return Stream(
            obs=np.asarray(data["obs"], np.uint8),
            reward=np.asarray(data["reward"], np.float32),
            events=np.asarray(data["events"], np.int32).reshape(-1, 4),
            gamma=float(data["gamma"]),
        )


def save_binary(stream: Stream, path: Path) -> None:
    """Write the stream in the flat little-endian layout plastax-cpp reads.

    Layout: magic "IMPR", u32 version, u32 steps, u32 n_obs, f32 gamma,
    u32 n_events; then steps*n_obs uint8 frames (row-major), steps float32
    rewards, n_events * 4 int32 event rows.
    """
    steps, n_obs = stream.obs.shape
    header = struct.pack(
        "<4sIIIfI",
        _BIN_MAGIC,
        _BIN_VERSION,
        steps,
        n_obs,
        stream.gamma,
        stream.events.shape[0],
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(stream.obs, np.uint8).tobytes())
        f.write(np.ascontiguousarray(stream.reward, "<f4").tobytes())
        f.write(np.ascontiguousarray(stream.events, "<i4").tobytes())


def load_binary(path: Path) -> Stream:
    """Read a stream written by `save_binary`."""
    with open(path, "rb") as f:
        magic, version, steps, n_obs, gamma, n_events = struct.unpack(
            "<4sIIIfI", f.read(24)
        )
        if magic != _BIN_MAGIC or version != _BIN_VERSION:
            raise ValueError(f"{path}: not a version-{_BIN_VERSION} stream file")
        obs = np.frombuffer(f.read(steps * n_obs), np.uint8).reshape(steps, n_obs)
        reward = np.frombuffer(f.read(4 * steps), "<f4").astype(np.float32)
        events = (
            np.frombuffer(f.read(16 * n_events), "<i4").astype(np.int32).reshape(-1, 4)
        )
    return Stream(obs=obs.copy(), reward=reward, events=events, gamma=float(gamma))


# --- cli ------------------------------------------------------------------


def _parse(argv: list[str] | None) -> tuple[StreamConfig, Path | None]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--out", type=Path, default=None, help="output stem (.npz and .bin are written)"
    )
    p.add_argument("--steps", type=int, default=StreamConfig.steps)
    p.add_argument("--seed", type=int, default=StreamConfig.seed)
    p.add_argument("--n-obs", type=int, default=StreamConfig.n_obs)
    p.add_argument("--active-bits", type=int, default=StreamConfig.active_bits)
    p.add_argument("--gamma", type=float, default=StreamConfig.gamma)
    p.add_argument("--jitter", type=float, default=StreamConfig.jitter)
    p.add_argument(
        "--background-jitter", type=float, default=StreamConfig.background_jitter
    )
    p.add_argument(
        "--background-drift", type=float, default=StreamConfig.background_drift
    )
    p.add_argument("--sound-len", type=int, default=StreamConfig.sound_len)
    p.add_argument(
        "--gap", type=int, nargs=2, default=StreamConfig.gap, metavar=("LO", "HI")
    )
    p.add_argument(
        "--delay", type=int, nargs=2, default=StreamConfig.delay, metavar=("LO", "HI")
    )
    a = p.parse_args(argv)
    cfg = StreamConfig(
        n_obs=a.n_obs,
        active_bits=a.active_bits,
        steps=a.steps,
        gamma=a.gamma,
        seed=a.seed,
        jitter=a.jitter,
        background_jitter=a.background_jitter,
        background_drift=a.background_drift,
        sound_len=a.sound_len,
        gap=(a.gap[0], a.gap[1]),
        delay=(a.delay[0], a.delay[1]),
    )
    return cfg, a.out


def main(argv: list[str] | None = None) -> None:
    """Generate a stream, write it, and print its event table."""
    cfg, out = _parse(argv)
    stream = generate(cfg)
    ret = discounted_return(stream.reward, stream.gamma)
    print(
        f"stream: {cfg.steps} steps, {cfg.n_obs} bits, {stream.events.shape[0]} events"
    )
    for sound, t_on, t_off, t_r in stream.events[:10]:
        print(
            f"  sound {sound} on {t_on:>6d} off {t_off:>6d} reward {t_r:>6d} "
            f"({stream.reward[t_r]:+.0f}); return at onset {ret[t_on]:+.3f}"
        )
    if stream.events.shape[0] > 10:
        print("  ...")
    # The best learnable prediction passes the shape criterion, the true
    # return (nonzero before the cue) and a silent predictor both fail it.
    _, perfect = shape_metrics(learnable_return(stream), stream)
    _, oracle = shape_metrics(ret, stream)
    _, silent = shape_metrics(np.zeros_like(ret), stream)
    assert perfect["pass"] == 1.0 and perfect["unrewarded_quiet"] == 1.0, perfect
    assert oracle["pre_quiet"] == 0.0, oracle
    assert silent["pass"] == 0.0, silent
    print(
        "shape criterion on the best learnable prediction:\n" + format_summary(perfect)
    )
    if out is not None:
        save_npz(stream, out.with_suffix(".npz"))
        save_binary(stream, out.with_suffix(".bin"))
        print(f"wrote {out.with_suffix('.npz')} and {out.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
