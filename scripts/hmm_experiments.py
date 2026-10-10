"""M2: HMMs on melody notes (one symbol per note, pitch class only)

Compares on the same note sequences, bits per note on test songs
  Markov chains of order 0 to 4 (Witten-Bell)
  supervised chord HMM: hidden states = annotated chords, plain and autoregressive emissions
  unsupervised HMMs trained with Baum-Welch, plain and autoregressive, several K, 3 seeds

Then asks whether the learned states recover the chords:
  map each learned state to the chord it co-occurs with most on train,
  then measure chord accuracy on test; compared with the best guess from the current note alone
  and with the supervised chord HMM

Per-note chord accuracy uses posterior decoding (most likely state at each note),
which maximizes expected correct notes; Viterbi (best single path) over-smooths here

Baum-Welch stops when the gain falls below 1e-4 nats per note (on K=8: 102 iterations,
train 2.388 bits vs 2.384 at full convergence after 200)

Writes outputs/hmm_results.csv and outputs/hmm_<name>.npz (model parameters)
Usage: python scripts/hmm_experiments.py [path to POP909 folder] [--quick] [--orders=0,1]
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from hmmusic.frames import N_CHORDS, note_sequences, split_songs, to_frames
from hmmusic.hmm import HMM
from hmmusic.markov import MarkovChain
from hmmusic.pop909 import load_song, song_folders

args = [a for a in sys.argv[1:] if not a.startswith("--")]
quick = "--quick" in sys.argv
root = args[0] if args else "../POP909-Dataset/POP909"
V = 12
SEEDS = [0] if quick else [0, 1, 2]
GRID = [(0, 8), (0, 25)] if quick else [(0, 8), (0, 25), (0, 50), (1, 4), (1, 8), (1, 25), (1, 50)]
MAX_ITER = 20 if quick else 300
# pseudo counts in the M step (MAP-EM): without them an autoregressive model gives probability 0
# to note pairs never seen in training, and one such pair in a test song makes its likelihood 0
ALPHA = 0.1
# optional --orders=0 or --orders=1 to run only part of the grid (for example on two terminals)
for a in sys.argv[1:]:
    if a.startswith("--orders="):
        keep = {int(o) for o in a.split("=")[1].split(",")}
        GRID = [(o, k) for o, k in GRID if o in keep]

frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
split = dict(zip(["train", "val", "test"], split_songs(list(frames), seed=0)))
data = {name: [s for i in ids for s in note_sequences(frames[i])] for name, ids in split.items()}
X = {name: [s[0] for s in seqs] for name, seqs in data.items()}
Z = {name: [s[1] for s in seqs] for name, seqs in data.items()}
print("notes train/val/test:", "/".join(str(sum(map(len, X[n]))) for n in ("train", "val", "test")))
print(f"uniform guess: {np.log2(V):.3f} bits per note\n")

out = Path("outputs")
out.mkdir(exist_ok=True)
rows = []


def report(name, val_bits, test_bits, chord_acc=None, extra=""):
    acc = f"{100 * chord_acc:5.1f}%" if chord_acc is not None else "     "
    print(f"{name:<28} val {val_bits:.3f}  test {test_bits:.3f}  chord acc {acc}  {extra}")
    rows.append({"model": name, "val_bits": val_bits, "test_bits": test_bits, "chord_acc": chord_acc})


def chord_accuracy(predicted: list[np.ndarray], truth: list[np.ndarray]) -> float:
    return float((np.concatenate(predicted) == np.concatenate(truth)).mean())


def state_to_chord_map(model: HMM) -> np.ndarray:
    """For each learned state, the chord it co-occurs with most often on train"""
    counts = np.zeros((model.n_states, N_CHORDS))
    for x, z in zip(X["train"], Z["train"]):
        np.add.at(counts, (model.posterior_decode(x), z), 1)
    return counts.argmax(axis=1)


# ---- baselines ----

print("-- baselines --")
for order in range(5):
    smoothing = "witten_bell" if order else "additive"
    m = MarkovChain(V, order, 1.0, smoothing).fit(X["train"])
    report(f"markov order {order}", m.bits_per_symbol(X["val"]), m.bits_per_symbol(X["test"]))

# chord from the current note only: most frequent chord for each pitch class on train
table = np.zeros((V, N_CHORDS))
for x, z in zip(X["train"], Z["train"]):
    np.add.at(table, (x, z), 1)
best_chord = table.argmax(axis=1)
acc = chord_accuracy([best_chord[x] for x in X["test"]], Z["test"])
print(f"{'chord from current note':<28} chord acc {100 * acc:5.1f}%")
rows.append({"model": "chord from current note", "chord_acc": acc})
majority = np.bincount(np.concatenate(Z["train"]), minlength=N_CHORDS).argmax()
acc = chord_accuracy([np.full(len(z), majority) for z in Z["test"]], Z["test"])
print(f"{'always the most common chord':<28} chord acc {100 * acc:5.1f}%")
rows.append({"model": "majority chord", "chord_acc": acc})

for order in (0, 1):
    m = HMM.fit_supervised(X["train"], Z["train"], N_CHORDS, V, alpha=0.1, emission_order=order)
    acc = chord_accuracy([m.posterior_decode(x) for x in X["test"]], Z["test"])
    report(f"chord hmm (supervised) ar{order}", m.bits_per_symbol(X["val"]), m.bits_per_symbol(X["test"]), acc)
    np.savez(out / f"hmm_chords_ar{order}.npz", pi=m.pi, A=m.A, B=m.B)

# ---- unsupervised ----

print("\n-- Baum-Welch --")
for order, K in GRID:
    for seed in SEEDS:
        t0 = time.time()
        m = HMM.random(K, V, np.random.default_rng(seed), emission_order=order)
        history = m.baum_welch(X["train"], n_iter=MAX_ITER, tol=1e-4, alpha=ALPHA)
        mapping = state_to_chord_map(m)
        acc = chord_accuracy([mapping[m.posterior_decode(x)] for x in X["test"]], Z["test"])
        name = f"bw ar{order} K={K} seed {seed}"
        report(name, m.bits_per_symbol(X["val"]), m.bits_per_symbol(X["test"]), acc,
               f"({len(history)} iter, {time.time() - t0:.0f}s)")
        np.savez(out / f"hmm_bw_ar{order}_K{K}_seed{seed}.npz", pi=m.pi, A=m.A, B=m.B)

suffix = "".join(a.replace("--orders=", "_ar").replace(",", "") for a in sys.argv[1:] if a.startswith("--orders="))
with open(out / f"hmm_results{suffix}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "val_bits", "test_bits", "chord_acc"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nresults written to {out}/hmm_results{suffix}.csv")
