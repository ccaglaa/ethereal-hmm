"""The two figures of the README

  figures/pianoroll.png        a generated piece (seed 1, AABA): lead notes, pad chords, sections
  figures/readme_results.png   chord tones and repetition: real songs vs generated vs control

Needs outputs/generation_eval.json, written by scripts/evaluate_generation.py
Usage: python scripts/make_readme_figures.py [--root=../POP909-Dataset/POP909]
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hmmusic.frames import N_CHORDS, beat_chords, chord_name, note_sequences, onset_phases, split_songs, to_frames  # noqa: E402
from hmmusic.generate import RhythmModel, generate_form  # noqa: E402
from hmmusic.hmm import HMM  # noqa: E402
from hmmusic.markov import MarkovChain  # noqa: E402
from hmmusic.pop909 import load_song, song_folders  # noqa: E402
from hmmusic.synth import merge_chords, pad_notes  # noqa: E402

PINK, VIOLET, PALE = "#e0447e", "#7b68ee", "#f4c2d5"  # pink + violet checked for color vision deficiency
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#f6dde7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
root = Path(opts.get("root", "../POP909-Dataset/POP909"))
out = Path("figures")
out.mkdir(exist_ok=True)


def piano_roll() -> None:
    frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
    train_ids, _, _ = split_songs(list(frames), seed=0)
    notes = [s for i in train_ids for s in note_sequences(frames[i])]
    rhythm = RhythmModel().fit([p for i in train_ids for p in onset_phases(frames[i])], [s[2] for s in notes])
    chain = MarkovChain(N_CHORDS, 4, 1.0, "witten_bell").fit([c for i in train_ids for c in beat_chords(frames[i])])
    hmm = HMM.fit_supervised([s[0] for s in notes], [s[1] for s in notes], N_CHORDS, 12, 0.1, 1)
    # same random stream as `python scripts/generate.py` for seed 1 (tempo drawn first), so the
    # figure shows exactly the piece in audio/ethereal_seed1_*.mp3
    rng = np.random.default_rng(1)
    rng.integers(99, 137)
    piece, labels = generate_form(rhythm, chain, hmm, rng)

    beats = 4.0  # x axis in bars: one beat = 1 / 4 bar, bpm does not matter here
    lead = piece.lead_notes(bpm=60)  # at 60 bpm one beat = 1 s, so seconds / 4 = bars
    chords = piece.chord_segments(bpm=60)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for n in pad_notes(chords, bass=False):
        ax.add_patch(plt.Rectangle((n.start / beats, n.pitch - 0.4), (n.end - n.start) / beats, 0.8,
                                   color=VIOLET, alpha=0.35, lw=0))
    for n in lead:
        ax.add_patch(plt.Rectangle((n.start / beats, n.pitch - 0.42), (n.end - n.start) / beats - 0.02, 0.84,
                                   color=PINK, lw=0))
    n_bars = len(piece.beat_chords) // 4
    top = max(n.pitch for n in lead) + 4
    for i, label in enumerate(labels):  # sections
        x0 = 4 * i
        if i:
            ax.axvline(x0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.text(x0 + 2, top + 1.5, label, ha="center", va="bottom", color=INK, fontsize=12, fontweight="bold")
    for start, end, symbol in merge_chords(chords):  # chord names under the roll
        if end - start >= 2:  # name chords lasting 2 beats or more, to avoid clutter
            ax.text((start + end) / 2 / beats, 50.2, chord_name(symbol), ha="center", va="bottom",
                    color=MUTED, fontsize=8)
    ax.set_xlim(0, n_bars)
    ax.set_ylim(49.5, top + 4)
    ax.set_xticks(range(0, n_bars + 1, 2))
    ax.set_xlabel("bar")
    yt = [p for p in range(52, top + 1) if p % 12 in (0, 7)]
    ax.set_yticks(yt, [f"{NOTE_NAMES[p % 12]}{p // 12 - 1}" for p in yt])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [matplotlib.patches.Patch(color=PINK, label="lead: the observed notes"),
               matplotlib.patches.Patch(color=VIOLET, alpha=0.35, label="pad: the hidden chords, voiced")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "pianoroll.png", dpi=160)
    plt.close(fig)


def results() -> None:
    data = json.loads(Path("outputs/generation_eval.json").read_text())
    rows = [  # (label, key in the json, bar color): violet = reference, pink = our final model
        ("Real songs (test set)", "test songs", VIOLET),
        ("Ours, AABA form", "chord HMM + AABA form", PINK),
        ("Ours, free sampling", "chord HMM (ours)", PALE),
        ("Control: notes ignore chords", "Markov 2, no chords", PALE),
    ]
    metrics = [("chord tone", "Notes that belong to the current chord"),
               ("repeated", "8-note patterns already heard in the piece")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (key, title) in zip(axes, metrics):
        values = [100 * data[k][key] for _, k, _ in rows]
        colors = [c for _, _, c in rows]
        ax.barh(y, values, height=0.56, color=colors)
        for yi, v in zip(y, values):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=8)
        ax.set_xlim(0, max(values) * 1.25 + 1)
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [label for label, _, _ in rows], color=INK)
    fig.tight_layout()
    fig.savefig(out / "readme_results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    piano_roll()
    results()
    print(f"figures written to {out}/")
