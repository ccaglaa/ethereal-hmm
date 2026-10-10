"""What did a trained HMM learn? One line per hidden state, measured on test songs

Each test note is assigned to its most likely state (posterior decoding); then per state
  share       fraction of test notes in this state
  stay        A[k, k], probability of staying in the state
  notes       most frequent pitch classes
  up / down / same   direction of the step from the previous note
  |step|      mean size of the step in semitones (folded to -6..+5)
  chords      annotated chords most often present at those notes

Usage: python scripts/inspect_states.py outputs/hmm_bw_ar1_K8_seed1.npz [path to POP909 folder]
"""

import sys

import numpy as np

from hmmusic.frames import N_CHORDS, PITCH_NAMES, chord_name, note_sequences, split_songs, to_frames
from hmmusic.hmm import HMM
from hmmusic.pop909 import load_song, song_folders

model_path = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "../POP909-Dataset/POP909"
params = np.load(model_path)
model = HMM(params["pi"], params["A"], params["B"])

frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
_, _, test_ids = split_songs(list(frames), seed=0)
test = [s for i in test_ids for s in note_sequences(frames[i])]

K = model.n_states
pitches = np.zeros((K, 12))
chords = np.zeros((K, N_CHORDS))
steps: list[list[int]] = [[] for _ in range(K)]
for x, z, _ in test:
    states = model.posterior_decode(x)
    step = (np.diff(x) + 6) % 12 - 6  # smallest signed step, for example B -> C is +1 not -11
    np.add.at(pitches, (states, x), 1)
    np.add.at(chords, (states, z), 1)
    for k, s in zip(states[1:], step):
        steps[k].append(int(s))

total = pitches.sum()
print(f"{model_path}: K={K}, emission order {model.emission_order}, {int(total)} test notes\n")
print("state share stay | notes                    | up   down same |step| | chords")
for k in np.argsort(-pitches.sum(1)):
    n = pitches[k].sum()
    if n == 0:
        continue
    top = " ".join(f"{PITCH_NAMES[i]:<2} {pitches[k, i] / n:.2f}" for i in np.argsort(-pitches[k])[:4])
    s = np.array(steps[k]) if steps[k] else np.zeros(1)
    top_chords = " ".join(chord_name(c) for c in np.argsort(-chords[k])[:3])
    print(
        f"{k:>5} {n / total:.2f}  {model.A[k, k]:.2f} | {top} | {np.mean(s > 0):.2f} {np.mean(s < 0):.2f} "
        f"{np.mean(s == 0):.2f} {np.mean(np.abs(s)):5.1f} | {top_chords}"
    )
