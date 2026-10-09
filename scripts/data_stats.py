"""Print basic facts about POP909 melodies, chords, beats and keys

Usage: python scripts/data_stats.py [path to POP909 folder]
"""

import sys
from collections import Counter

import numpy as np

from hmmusic.pop909 import load_song, song_folders

root = sys.argv[1] if len(sys.argv) > 1 else "../POP909-Dataset/POP909"
songs = [load_song(f) for f in song_folders(root)]
print(f"songs: {len(songs)}")

n_notes = [len(s.melody) for s in songs]
print(f"melody notes: {sum(n_notes)} total, median {np.median(n_notes):.0f} per song")

pitches = np.array([n.pitch for s in songs for n in s.melody])
low, mid, high = np.percentile(pitches, [5, 50, 95])
print(f"melody pitch: 5th pct {low:.0f}, median {mid:.0f}, 95th pct {high:.0f} (60 = middle C)")

beat = np.concatenate([np.diff(s.grid) for s in songs])
print(f"beat length: median {np.median(beat):.3f} s, i.e. ar. {60 / np.median(beat):.0f} bpm")

# where do note onsets fall inside a beat, in quarters of a beat
phase = []
for s in songs:
    i = np.searchsorted(s.grid, [n.start for n in s.melody], side="right") - 1
    ok = (i >= 0) & (i < len(s.grid) - 1)
    starts = np.array([n.start for n in s.melody])[ok]
    frac = (starts - s.grid[i[ok]]) / (s.grid[i[ok] + 1] - s.grid[i[ok]])
    phase.extend(np.round(frac * 4) % 4)
counts = Counter(int(p) for p in phase)
total = sum(counts.values())
print("onset position in the beat: " + ", ".join(f"{k}/4: {100 * counts[k] / total:.1f}%" for k in range(4)))

overlaps = sum(1 for s in songs for a, b in zip(s.melody, s.melody[1:]) if b.start < a.end - 1e-3)
print(f"overlapping consecutive melody notes: {overlaps} ({100 * overlaps / sum(n_notes):.1f}%)")

print(f"songs with a single key: {sum(len(s.keys) == 1 for s in songs)}")
print("first key mode:", dict(Counter(s.keys[0].label.split(":")[1] for s in songs)))

qualities = Counter(c.label.split(":")[1] if ":" in c.label else c.label for s in songs for c in s.chords)
print(f"chord qualities: {len(qualities)}, most common:", qualities.most_common(8))
