"""The synthetic audio-like stream of examples/imprinting_stream.py.

Pins the invariants the two drivers (plastax, plastax-cpp) and the GVF-shape
criterion rely on: fixed active-bit count per frame, event timing inside the
configured ranges, determinism per seed, the return recursion, the file
round-trips, and that the shape criterion accepts exactly the best learnable
prediction. The example is loaded by file path (examples/ is not a package).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ims = _load_example("imprinting_stream")

_CFG = ims.StreamConfig(steps=6000, seed=3)


@pytest.fixture(scope="module")
def stream() -> object:
    return ims.generate(_CFG)


def test_every_frame_has_exactly_active_bits_ones(stream: object) -> None:
    counts = stream.obs.sum(axis=1)
    assert stream.obs.dtype == np.uint8
    assert np.all(counts == _CFG.active_bits)


def test_events_respect_gap_delay_and_reward_placement(stream: object) -> None:
    ev = stream.events
    assert ev.shape[1] == 4 and ev.shape[0] > 5
    prev_off = 0
    for sound, t_on, t_off, t_r in ev:
        assert t_off - t_on == _CFG.sound_len
        assert _CFG.gap[0] <= t_on - prev_off <= _CFG.gap[1]
        assert _CFG.delay[0] <= t_r - t_off <= _CFG.delay[1]
        assert t_r < _CFG.steps
        assert stream.reward[t_r] == np.float32(_CFG.rewards[sound])
        prev_off = t_off
    # Rewards live only on event reward steps.
    assert np.count_nonzero(stream.reward) == np.count_nonzero(
        [_CFG.rewards[s] for s in ev[:, ims.EVENT_SOUND]]
    )


def test_sound_frames_replay_the_template_with_bounded_jitter(stream: object) -> None:
    ev = stream.events
    by_sound: dict[int, list[np.ndarray]] = {}
    for sound, t_on, t_off, _ in ev:
        by_sound.setdefault(int(sound), []).append(stream.obs[t_on:t_off])
    for replays in by_sound.values():
        if len(replays) < 2:
            continue
        a, b = replays[0], replays[1]
        # Two replays of one sound share most bits of every frame: with
        # jitter 0.1 on 8 bits, an expected 0.8 bits move per frame per replay.
        overlap = (a & b).sum(axis=1)
        assert overlap.mean() >= _CFG.active_bits - 2.5


def test_generation_is_deterministic_per_seed() -> None:
    a = ims.generate(_CFG)
    b = ims.generate(_CFG)
    c = ims.generate(ims.StreamConfig(steps=6000, seed=4))
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.events, b.events)
    assert not np.array_equal(a.obs, c.obs)


def test_discounted_return_recursion() -> None:
    reward = np.zeros((10,), np.float32)
    reward[7] = 1.0
    ret = ims.discounted_return(reward, 0.5)
    # G_t = r_{t+1} + gamma G_{t+1}: the reward on 7 pays the prediction on 6.
    assert ret[6] == pytest.approx(1.0)
    assert ret[5] == pytest.approx(0.5)
    assert ret[0] == pytest.approx(0.5**6)
    assert np.all(ret[7:] == 0.0)


def test_learnable_return_is_zero_outside_cue_windows(stream: object) -> None:
    target = ims.learnable_return(stream)
    ret = ims.discounted_return(stream.reward, stream.gamma)
    for _, t_on, _, t_r in stream.events:
        if stream.reward[t_r] != 0.0:
            assert np.array_equal(target[t_on:t_r], ret[t_on:t_r])
        else:
            assert np.all(target[t_on:t_r] == 0.0)
        assert np.all(target[t_r : t_r + 5] == 0.0)
        assert np.all(target[t_on - 5 : t_on] == 0.0)


def test_shape_criterion_accepts_the_learnable_target_only(stream: object) -> None:
    target = ims.learnable_return(stream)
    _, perfect = ims.shape_metrics(target, stream)
    assert perfect["pass"] == 1.0
    assert perfect["unrewarded_quiet"] == 1.0
    _, silent = ims.shape_metrics(np.zeros_like(target), stream)
    assert silent["pass"] == 0.0
    assert silent["onset"] == 0.0
    # A predictor that only fires while the sound is audible (thesis Figure
    # 9.4) passes onset but fails the sustained window.
    audible = np.zeros_like(target)
    for _, t_on, t_off, _ in stream.events:
        audible[t_on:t_off] = target[t_on:t_off]
    rows, summary = ims.shape_metrics(audible, stream)
    assert summary["onset"] == 1.0
    assert summary["pass"] == 0.0
    assert all(set(ims.SHAPE_COLUMNS) == set(row) for row in rows)


def test_file_round_trips(stream: object, tmp_path: Path) -> None:
    ims.save_npz(stream, tmp_path / "s.npz")
    ims.save_binary(stream, tmp_path / "s.bin")
    for loaded in (
        ims.load_npz(tmp_path / "s.npz"),
        ims.load_binary(tmp_path / "s.bin"),
    ):
        assert np.array_equal(loaded.obs, stream.obs)
        assert np.array_equal(loaded.reward, stream.reward)
        assert np.array_equal(loaded.events, stream.events)
        assert loaded.gamma == pytest.approx(stream.gamma)
    size = (tmp_path / "s.bin").stat().st_size
    assert (
        size
        == 24 + stream.obs.size + 4 * stream.reward.size + 16 * stream.events.shape[0]
    )
