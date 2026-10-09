"""Markov chain baselines on melody frames: held-out bits per frame, additive vs Witten-Bell smoothing

Additive smoothing alpha is chosen on the validation songs, the score is reported on test songs
Also writes a few sampled melodies as MIDI to outputs/

Usage: python scripts/markov_baselines.py [path to POP909 folder]
"""

import sys
from pathlib import Path

import numpy as np

from hmmusic.decode import obs_to_notes
from hmmusic.frames import N_OBS, OBS_NAMES, phrases, split_songs, to_frames
from hmmusic.markov import MarkovChain
from hmmusic.midi import Track, write_midi
from hmmusic.pop909 import load_song, song_folders

root = sys.argv[1] if len(sys.argv) > 1 else "../POP909-Dataset/POP909"
frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
train_ids, val_ids, test_ids = split_songs(list(frames), seed=0)


def melody_phrases(ids):
    return [obs for i in ids for obs, _ in phrases(frames[i])]


train, val, test = melody_phrases(train_ids), melody_phrases(val_ids), melody_phrases(test_ids)
print(f"songs train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
print(f"phrases: {len(train)}/{len(val)}/{len(test)}, frames: {sum(map(len, train))}/{sum(map(len, val))}/{sum(map(len, test))}")
print(f"uniform guess: {np.log2(N_OBS):.3f} bits per frame\n")

print("smoothing     order  alpha   val bits  test bits  perplexity")
best = {}
for smoothing, orders in [("additive", range(5)), ("witten_bell", range(1, 6))]:
    for order in orders:
        # alpha only matters for additive; for Witten-Bell it is a unigram floor with no effect here
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0] if smoothing == "additive" else [1.0]
        scores = {a: MarkovChain(N_OBS, order, a, smoothing).fit(train).bits_per_symbol(val) for a in alphas}
        alpha = min(scores, key=scores.get)
        model = MarkovChain(N_OBS, order, alpha, smoothing).fit(train)
        test_bits = model.bits_per_symbol(test)
        best[smoothing, order] = model
        shown = alpha if smoothing == "additive" else "-"
        print(f"{smoothing:<12}  {order:>5}  {shown:>5}  {scores[alpha]:>8.3f}  {test_bits:>9.3f}  {2 ** test_bits:>10.2f}")

out = Path("outputs")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
for smoothing, order in [("additive", 1), ("additive", 4), ("witten_bell", 4)]:
    for k in range(3):
        obs = best[smoothing, order].sample(128, rng)  # 128 frames = 8 bars of 4/4
        name = f"markov_{smoothing}_order{order}_sample{k}.mid"
        write_midi(out / name, [Track("MELODY", obs_to_notes(obs))], bpm=80)
    print(f"\n{smoothing} order {order} sample:", " ".join(OBS_NAMES[x] for x in obs[:48]))
print(f"\nMIDI samples written to {out}/")
