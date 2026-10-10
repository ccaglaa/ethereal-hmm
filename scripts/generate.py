"""Generate ethereal pieces: rhythm model + chord per beat + notes from the ar1 chord HMM

Trains the three models on the training songs, picks the chord chain order on validation songs,
then writes outputs/ethereal_seed<k>.wav and .mid (tracks LEAD and PAD, open the MIDI in GarageBand)

Usage: python scripts/generate.py [--seeds=0,1,2] [--bars=16] [--bpm=60] [--root=../POP909-Dataset/POP909]
"""

import sys
from pathlib import Path

import numpy as np

from hmmusic.frames import N_CHORDS, N_OBS, beat_chords, chord_name, note_sequences, onset_phases, split_songs, to_frames
from hmmusic.generate import RhythmModel, generate
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.midi import Track, write_midi
from hmmusic.pop909 import load_song, song_folders
from hmmusic.synth import pad_notes, render, write_wav

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
root = Path(opts.get("root", "../POP909-Dataset/POP909"))
seeds = [int(s) for s in opts.get("seeds", "0,1,2").split(",")]
n_bars = int(opts.get("bars", 16))
bpm = float(opts.get("bpm", 60))

frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
train_ids, val_ids, _ = split_songs(list(frames), seed=0)

# rhythm
phases = [p for i in train_ids for p in onset_phases(frames[i])]
gaps = [g for i in train_ids for _, _, g in note_sequences(frames[i])]
rhythm = RhythmModel().fit(phases, gaps)

# chords per beat, order chosen on validation songs
# orders above 4 do not fit in memory: the table is dense, (25 + 1)^order x 25 entries
train_chords = [c for i in train_ids for c in beat_chords(frames[i])]
val_chords = [c for i in val_ids for c in beat_chords(frames[i])]
scores = {k: MarkovChain(N_CHORDS, k, 1.0, "witten_bell").fit(train_chords).bits_per_symbol(val_chords) for k in range(1, 5)}
order = min(scores, key=scores.get)
chord_chain = MarkovChain(N_CHORDS, order, 1.0, "witten_bell").fit(train_chords)
print("chord chain, val bits per beat by order:", {k: round(v, 3) for k, v in scores.items()}, "-> order", order)

# notes: emission table of the ar1 chord HMM (pitch classes only, 12 symbols)
train_notes = [s for i in train_ids for s in note_sequences(frames[i])]
note_hmm = HMM.fit_supervised([s[0] for s in train_notes], [s[1] for s in train_notes], N_CHORDS, 12, 0.1, 1)

out = Path("outputs")
out.mkdir(exist_ok=True)
for seed in seeds:
    piece = generate(rhythm, chord_chain, note_hmm, n_bars, np.random.default_rng(seed))
    lead = piece.lead_notes(bpm)
    chords = piece.chord_segments(bpm)
    write_midi(out / f"ethereal_seed{seed}.mid", [Track("LEAD", lead), Track("PAD", pad_notes(chords))], bpm=bpm)
    write_wav(out / f"ethereal_seed{seed}.wav", render(lead, chords, bpm, seed=seed))
    bars = [" ".join(chord_name(c) for c in piece.beat_chords[4 * b : 4 * b + 4]) for b in range(n_bars)]
    print(f"\nseed {seed}: {len(lead)} notes, {n_bars} bars at {bpm:.0f} bpm -> {out}/ethereal_seed{seed}.wav")
    print("  chords: " + " | ".join(bars[:8]))
