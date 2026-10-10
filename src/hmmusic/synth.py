"""A small ethereal synthesizer in NumPy: soft lead, detuned pad, delay, reverb, WAV output

Signal flow
  lead notes  -> soft additive tone with slow vibrato -> ping-pong delay --\\
  pad chords  -> 3 detuned voices per note, slow attack, long release ------+-> reverb -> soft clip -> WAV

Everything is additive synthesis (sums of sine harmonics), so there is no aliasing to worry about,
and the reverb is a convolution with a synthetic impulse response (decaying noise), done with the FFT
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hmmusic.midi import Note

SR = 44100


def midi_to_hz(pitch: float) -> float:
    """MIDI pitch -> frequency, A4 = MIDI 69 = 440 Hz, 12 semitones per octave"""
    return 440.0 * 2 ** ((pitch - 69) / 12)


@dataclass(frozen=True)
class Tone:
    """Recipe for one instrument

    harmonics  relative amplitude of harmonics 1, 2, 3 ...
    detune     offsets in cents, one oscillator per value (several = chorus like width)
    attack, release   seconds
    vibrato    depth in cents and rate in Hz, fading in over vibrato_delay seconds
    """

    harmonics: tuple[float, ...]
    detune: tuple[float, ...] = (0.0,)
    attack: float = 0.05
    release: float = 0.5
    vibrato_depth: float = 0.0
    vibrato_rate: float = 5.0
    vibrato_delay: float = 0.3


LEAD = Tone(harmonics=(1.0, 0.25, 0.08, 0.03), detune=(-4.0, 4.0), attack=0.08, release=1.2,
            vibrato_depth=12.0, vibrato_rate=4.5, vibrato_delay=0.35)
PAD = Tone(harmonics=tuple(1 / k**2.5 for k in range(1, 6)), detune=(-9.0, 0.0, 9.0), attack=1.6, release=2.5)  # dark: few, quiet harmonics
PLUCK = Tone(harmonics=(1.0, 0.4, 0.15, 0.06), detune=(-6.0, 6.0), attack=0.005, release=0.7)  # arpeggio notes


@dataclass(frozen=True)
class Preset:
    """Everything that shapes the sound, so the same notes can be rendered several ways

    pad_attack_beats   pad attack in beats, capped by the pad's own attack in seconds: at fast tempos
                       a fixed 1.6 s attack would still be fading in when the next chord comes
    shimmer            level of a copy of the lead one octave higher (0 = off), a glassy halo
    arpeggio           level of the chord tones played as 8th notes (0 = off), gives a pulse
    """

    lead: Tone = LEAD
    pad: Tone = PAD
    pad_level: float = 0.35
    pad_attack_beats: float = 1.5
    reverb_rt60: float = 4.5
    reverb_mix: float = 0.45
    delay_feedback: float = 0.45
    shimmer: float = 0.0
    arpeggio: float = 0.0


PRESETS = {
    # the M3 sound: soft lead, dark pad, long reverb
    "airy": Preset(),
    # darker and wider: softer lead, slower pad, longer and wetter reverb, more echoes
    "dreamy": Preset(
        lead=replace(LEAD, harmonics=(1.0, 0.12, 0.03), release=1.8, vibrato_depth=8.0),
        pad=replace(PAD, release=4.0), pad_level=0.45, pad_attack_beats=2.5,
        reverb_rt60=7.0, reverb_mix=0.6, delay_feedback=0.55),
    # brighter, bell like lead with an octave shimmer, lighter pad
    "glass": Preset(
        lead=replace(LEAD, harmonics=(1.0, 0.45, 0.25, 0.15, 0.08), release=1.5),
        pad_level=0.25, shimmer=0.35, reverb_mix=0.5),
    # for faster tempos: chord tones as 8th notes over a quieter pad, shorter reverb
    "pulse": Preset(pad_level=0.2, pad_attack_beats=1.0, reverb_rt60=3.0, reverb_mix=0.35,
                    delay_feedback=0.35, arpeggio=0.3),
}


def envelope(n_hold: int, n_release: int, n_attack: int) -> np.ndarray:
    """Attack (raised cosine) -> sustain at 1 -> release (exponential to ar. -60 dB)

    The note is held for n_hold samples; the release tail is added after it
    If the note is shorter than the attack, the release starts from the level reached
    """
    n = n_hold + n_release
    env = np.ones(n)
    a = min(n_attack, n_hold)
    if n_attack > 0:
        env[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / n_attack)
    level = env[n_hold - 1]  # level at the end of the hold: 1 if the attack finished, lower if not
    if n_release > 0:
        env[n_hold:] = level * np.exp(-6.9 * np.arange(n_release) / n_release)
    return env


def tone(pitch: float, duration: float, recipe: Tone, velocity: float = 0.8, sr: int = SR,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Render one note (mono), including its release tail"""
    rng = rng or np.random.default_rng(0)
    n_hold = max(1, int(duration * sr))
    n_release = int(recipe.release * sr)
    n = n_hold + n_release
    t = np.arange(n) / sr
    # vibrato in cents, fading in so the start of the note is steady
    fade = np.clip((t - recipe.vibrato_delay) / 0.5, 0.0, 1.0)
    cents = recipe.vibrato_depth * fade * np.sin(2 * np.pi * recipe.vibrato_rate * t)
    out = np.zeros(n)
    for detune in recipe.detune:
        f = midi_to_hz(pitch) * 2 ** ((cents + detune) / 1200)
        # phase = integral of frequency, so vibrato bends the pitch smoothly
        phase = 2 * np.pi * np.cumsum(f) / sr + rng.uniform(0, 2 * np.pi)
        for k, amp in enumerate(recipe.harmonics, start=1):
            if k * f.max() < sr / 2:  # skip harmonics above Nyquist
                out += amp * np.sin(k * phase)
    out /= len(recipe.detune) * sum(recipe.harmonics)
    return velocity * out * envelope(n_hold, n_release, int(recipe.attack * sr))


def render_notes(notes: list[Note], recipe: Tone, total: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Mix notes into a mono buffer of total seconds (tails past the end are cut)"""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(total * sr))
    for note in notes:
        y = tone(note.pitch, note.end - note.start, recipe, note.velocity / 127, sr, rng)
        i = int(note.start * sr)
        j = min(len(out), i + len(y))
        if i < len(out):
            out[i:j] += y[: j - i]
    return out


# ---- chords -> pad notes ----

def chord_pitch_classes(symbol: int) -> list[int]:
    """Chord symbol (root * 2 + minor) -> pitch classes of root, third, fifth"""
    root, minor = symbol // 2, symbol % 2
    return [root, (root + (3 if minor else 4)) % 12, (root + 7) % 12]


def voice_chord(pcs: list[int], previous: list[int] | None, low: int = 52, high: int = 72) -> list[int]:
    """Pick one pitch per pitch class in [low, high], moving as little as possible from previous

    This is voice leading: consecutive chords share or move by small steps instead of jumping
    """
    options = [[p for p in range(low, high + 1) if p % 12 == pc] for pc in pcs]
    best, best_cost = None, np.inf
    for combo in np.array(np.meshgrid(*options)).T.reshape(-1, len(pcs)):
        voicing = sorted(int(p) for p in combo)
        if voicing[-1] - voicing[0] > 12:  # keep it compact, within an octave
            continue
        if previous is None:
            cost = abs(np.mean(voicing) - (low + high) / 2)
        else:
            cost = sum(abs(a - b) for a, b in zip(voicing, sorted(previous)))
        if cost < best_cost:
            best, best_cost = voicing, cost
    return best


def merge_chords(chords: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Join consecutive segments with the same chord (the generator gives one segment per beat)"""
    merged: list[list] = []
    for start, end, symbol in chords:
        if merged and merged[-1][2] == symbol and abs(merged[-1][1] - start) < 1e-6:
            merged[-1][1] = end
        else:
            merged.append([start, end, symbol])
    return [tuple(m) for m in merged]


def pad_notes(chords: list[tuple[float, float, int]], velocity: int = 70, bass: bool = True) -> list[Note]:
    """(start, end, chord symbol) segments -> pad notes, consecutive equal chords merged

    Chord symbol 24 (no chord) plays nothing
    """
    notes, previous = [], None
    for start, end, symbol in merge_chords(chords):
        if symbol >= 24:
            continue
        voicing = voice_chord(chord_pitch_classes(symbol), previous)
        previous = voicing
        notes += [Note(start, end, p, velocity) for p in voicing]
        if bass:
            notes.append(Note(start, end, 36 + symbol // 2, velocity - 15))  # root, octave C2 to B2
    return notes


def arpeggio_notes(chords: list[tuple[float, float, int]], bpm: float, velocity: int = 70) -> list[Note]:
    """Chord tones as 8th notes going up and down (root third fifth octave fifth third ...)"""
    eighth = 30.0 / bpm
    notes, previous = [], None
    for start, end, symbol in merge_chords(chords):
        if symbol >= 24:
            continue
        voicing = voice_chord(chord_pitch_classes(symbol), previous, low=60, high=79)
        previous = voicing
        pattern = voicing + [voicing[0] + 12] + voicing[1:][::-1]  # up then down
        t, i = start, 0
        while t < end - 1e-9:
            notes.append(Note(t, min(t + eighth, end), pattern[i % len(pattern)], velocity))
            t, i = t + eighth, i + 1
    return notes


# ---- effects ----

def ping_pong_delay(mono: np.ndarray, delay: float, feedback: float = 0.45, repeats: int = 6,
                    sr: int = SR) -> np.ndarray:
    """Echoes alternating left and right, each quieter by feedback; returns stereo (n, 2)"""
    d = int(delay * sr)
    out = np.zeros((len(mono) + d * repeats, 2))
    out[: len(mono)] += mono[:, None]
    for r in range(1, repeats + 1):
        side = r % 2  # 1 = right, 0 = left, so echoes bounce
        out[r * d : r * d + len(mono), side] += feedback**r * mono
    return out[: len(mono)]


def reverb_impulse(seconds: float = 5.0, rt60: float = 4.5, predelay: float = 0.03, sr: int = SR,
                   seed: int = 0) -> np.ndarray:
    """Synthetic stereo impulse response: decaying noise, darker as it decays

    rt60 = time for the level to fall by 60 dB (a factor 1000); exp(-6.9 t / rt60) does that
    Darkening: a bright part that dies fast plus a smoothed (low passed) part that lasts
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = np.zeros((n, 2))
    for ch in range(2):  # independent noise left and right gives width
        bright = rng.standard_normal(n) * np.exp(-6.9 * t / (rt60 / 3))
        dark = rng.standard_normal(n)
        kernel = np.ones(16) / 16  # moving average = simple low pass
        dark = np.convolve(dark, kernel, mode="same") * np.exp(-6.9 * t / rt60)
        ir[:, ch] = 0.4 * bright + dark
    ir = np.vstack([np.zeros((int(predelay * sr), 2)), ir])
    return ir / np.sqrt((ir**2).sum(axis=0, keepdims=True))  # unit energy per channel


def convolve_fft(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Linear convolution of each channel with the FFT, output cut to len(x)"""
    n = len(x) + len(ir) - 1
    size = 1 << (n - 1).bit_length()  # next power of two
    out = np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0)
    return out[: len(x)]


def master(stereo: np.ndarray, peak: float = 0.89, fade: float = 2.0, sr: int = SR) -> np.ndarray:
    """Soft clip, fade in and out, then normalize to peak (0.89 = -1 dBFS)

    Normalizing last guarantees the peak even when the loudest moment falls inside a fade
    """
    y = np.tanh(1.2 * stereo / (np.abs(stereo).max() + 1e-12))
    n = min(int(fade * sr), len(y) // 2)
    ramp = np.linspace(0, 1, n)[:, None]
    y[:n] *= ramp
    y[len(y) - n :] *= ramp[::-1]
    return y * peak / (np.abs(y).max() + 1e-12)


def render(lead: list[Note], chords: list[tuple[float, float, int]], bpm: float, preset: Preset | None = None,
           tail: float = 5.0, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Full mix as stereo float in [-1, 1], shape (n, 2)"""
    preset = preset or PRESETS["airy"]
    beat = 60.0 / bpm
    end = max([n.end for n in lead] + [c[1] for c in chords] + [0.0]) + tail

    lead_dry = render_notes(lead, preset.lead, end, sr, seed)
    if preset.shimmer > 0:
        octave_up = [Note(n.start, n.end, n.pitch + 12, n.velocity) for n in lead]
        lead_dry = lead_dry + preset.shimmer * render_notes(octave_up, preset.lead, end, sr, seed + 2)
    lead_wide = ping_pong_delay(lead_dry, delay=0.75 * beat, feedback=preset.delay_feedback, sr=sr)  # dotted 8th

    pad_tone = replace(preset.pad, attack=min(preset.pad.attack, preset.pad_attack_beats * beat))
    pad = render_notes(pad_notes(chords), pad_tone, end, sr, seed + 1)
    dry = lead_wide + preset.pad_level * pad[:, None]
    if preset.arpeggio > 0:
        arp = render_notes(arpeggio_notes(chords, bpm), PLUCK, end, sr, seed + 3)
        dry = dry + preset.arpeggio * ping_pong_delay(arp, delay=0.75 * beat, feedback=0.3, sr=sr)

    wet = convolve_fft(dry, reverb_impulse(rt60=preset.reverb_rt60, seconds=max(5.0, 1.2 * preset.reverb_rt60),
                                           sr=sr, seed=seed))
    wet *= np.abs(dry).max() / (np.abs(wet).max() + 1e-12)  # bring the wet signal to a similar level
    return master((1 - preset.reverb_mix) * dry + preset.reverb_mix * wet, sr=sr)


def write_wav(path: str | Path, stereo: np.ndarray, sr: int = SR) -> None:
    """16 bit stereo WAV with the standard library"""
    data = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())  # rows are (left, right), so this is interleaved
