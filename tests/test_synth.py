import wave
from pathlib import Path

import numpy as np
import pytest

from hmmusic.midi import Note
from hmmusic.synth import (
    Tone,
    chord_pitch_classes,
    convolve_fft,
    envelope,
    midi_to_hz,
    pad_notes,
    ping_pong_delay,
    render,
    reverb_impulse,
    tone,
    voice_chord,
    write_wav,
)

SR = 8000  # low sample rate keeps the tests fast


@pytest.mark.parametrize("pitch, hz", [(69, 440.0), (81, 880.0), (57, 220.0), (60, 261.6256)])
def test_midi_to_hz(pitch, hz):
    assert midi_to_hz(pitch) == pytest.approx(hz, rel=1e-6)


def test_envelope_shape():
    env = envelope(n_hold=100, n_release=50, n_attack=20)
    assert len(env) == 150
    assert env[0] == 0.0
    assert env[20:100].min() == 1.0
    assert env[100] == pytest.approx(1.0)
    assert env[-1] < 0.002  # ar. -60 dB at the end of the release


def test_envelope_short_note_has_no_jump():
    env = envelope(n_hold=10, n_release=50, n_attack=40)
    assert np.abs(np.diff(env)).max() < 0.1


def test_tone_frequency():
    recipe = Tone(harmonics=(1.0,), attack=0.0, release=0.0)
    y = tone(69, 1.0, recipe, sr=SR)
    spectrum = np.abs(np.fft.rfft(y))
    assert np.fft.rfftfreq(len(y), 1 / SR)[spectrum.argmax()] == pytest.approx(440, abs=1.5)


def test_tone_skips_harmonics_above_nyquist():
    recipe = Tone(harmonics=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), attack=0.0, release=0.0)
    y = tone(69, 0.5, recipe, sr=SR)  # 10th harmonic = 4400 Hz > 4000 Hz Nyquist, must be skipped
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    assert spectrum[freqs > 1000].max() < 0.01 * spectrum.max()


@pytest.mark.parametrize("symbol, pcs", [(0, [0, 4, 7]), (1, [0, 3, 7]), (19, [9, 0, 4]), (14, [7, 11, 2])])
def test_chord_pitch_classes(symbol, pcs):
    assert chord_pitch_classes(symbol) == pcs


def test_voice_chord_range_and_compactness():
    v = voice_chord([0, 4, 7], None)
    assert all(52 <= p <= 72 for p in v)
    assert v[-1] - v[0] <= 12
    assert sorted(p % 12 for p in v) == [0, 4, 7]


def test_voice_leading_keeps_common_tones():
    c_major = [55, 60, 64]  # G3 C4 E4
    a_minor = voice_chord([9, 0, 4], c_major)  # shares C and E with C major
    assert 60 in a_minor and 64 in a_minor
    assert sum(abs(a - b) for a, b in zip(a_minor, c_major)) == 2  # G3 -> A3 only


def test_pad_notes_merges_and_skips_no_chord():
    notes = pad_notes([(0.0, 1.0, 0), (1.0, 2.0, 0), (2.0, 3.0, 24), (3.0, 4.0, 19)], bass=False)
    starts = sorted({(n.start, n.end) for n in notes})
    assert starts == [(0.0, 2.0), (3.0, 4.0)]
    assert len(notes) == 6


def test_ping_pong_delay():
    x = np.zeros(100)
    x[0] = 1.0
    y = ping_pong_delay(x, delay=10 / SR, feedback=0.5, repeats=3, sr=SR)
    assert y[0].tolist() == [1.0, 1.0]
    assert y[10].tolist() == [0.0, 0.5]  # first echo on the right
    assert y[20].tolist() == [0.25, 0.0]  # second on the left
    assert y[30].tolist() == [0.0, 0.125]


def test_reverb_impulse_energy_and_decay():
    ir = reverb_impulse(seconds=2.0, rt60=1.0, predelay=0.01, sr=SR)
    np.testing.assert_allclose((ir**2).sum(axis=0), 1.0)
    assert np.abs(ir[: int(0.01 * SR)]).max() == 0.0  # predelay is silent
    first = (ir[: SR // 2] ** 2).sum()
    last = (ir[-SR // 2 :] ** 2).sum()
    assert last < 1e-4 * first


def test_convolve_fft_matches_numpy():
    rng = np.random.default_rng(0)
    x, h = rng.standard_normal((50, 2)), rng.standard_normal((7, 2))
    y = convolve_fft(x, h)
    for ch in range(2):
        np.testing.assert_allclose(y[:, ch], np.convolve(x[:, ch], h[:, ch])[:50], atol=1e-10)


def test_write_wav_round_trip(tmp_path: Path):
    t = np.arange(SR) / SR
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), -0.25 * np.ones(SR)], axis=1)
    write_wav(tmp_path / "a.wav", stereo, sr=SR)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) == (2, 2, SR, SR)
        data = np.frombuffer(f.readframes(SR), "<i2").reshape(-1, 2) / 32767
    np.testing.assert_allclose(data, stereo, atol=1 / 32767)


def test_render_full_mix():
    lead = [Note(0.0, 0.5, 72, 90), Note(0.5, 1.5, 76, 90)]
    audio = render(lead, [(0.0, 1.5, 0)], bpm=70, tail=1.0, sr=SR)
    assert audio.shape == (int(2.5 * SR), 2)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() == pytest.approx(0.89, abs=1e-6)
