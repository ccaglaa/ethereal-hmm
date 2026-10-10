"""Render a real POP909 song through the ethereal synth: its melody as the lead, its chords as the pad

A reference for the sound, on music we know is good, before generating anything
Writes outputs/song_<id>.wav

Usage: python scripts/render_song.py [song id, default the slowest test song] [--seconds=45] [--slow=1.25]
       [--root=../POP909-Dataset/POP909]
"""

import sys
from pathlib import Path

import numpy as np

from hmmusic.frames import chord_symbol, split_songs, to_frames
from hmmusic.midi import Note
from hmmusic.pop909 import load_song, song_folders
from hmmusic.synth import render, write_wav

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = Path(opts.get("root", "../POP909-Dataset/POP909"))
seconds = float(opts.get("seconds", 45))
slow = float(opts.get("slow", 1.25))  # 1.25 = 25% slower, more floating

if args:
    song_id = args[0]
else:
    frames = {f.song_id: f for f in (to_frames(load_song(d)) for d in song_folders(root)) if f is not None}
    # default: the slowest test song (longest median beat), slow songs suit the sound best
    test_ids = split_songs(list(frames), seed=0)[2]
    song_id = max(sorted(test_ids), key=lambda i: np.median(np.diff(load_song(root / i).grid)))
song = load_song(root / song_id)

t0 = song.melody[0].start - 0.5  # start just before the first sung note
lead = [
    Note((n.start - t0) * slow, (n.end - t0) * slow, n.pitch, n.velocity)
    for n in song.melody
    if t0 <= n.start < t0 + seconds
]
chords = [
    ((max(c.start, t0) - t0) * slow, (min(c.end, t0 + seconds) - t0) * slow, chord_symbol(c.label, 0))
    for c in song.chords
    if c.end > t0 and c.start < t0 + seconds
]
bpm = 60 / np.median(np.diff(song.grid)) / slow

audio = render(lead, chords, bpm)
out = Path("outputs")
out.mkdir(exist_ok=True)
path = out / f"song_{song_id}.wav"
write_wav(path, audio)
print(f"song {song_id}: {len(lead)} notes, {len(chords)} chord segments, {bpm:.0f} bpm, {len(audio) / 44100:.1f} s -> {path}")
