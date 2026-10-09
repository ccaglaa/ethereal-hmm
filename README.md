# ethereal-hmm

Generate ethereal synth melodies with a hidden Markov model written from scratch in NumPy.

The hidden state is the chord, the observation is the melody note. When generating, the hidden chord
sequence is rendered as a soft synth pad and the observed notes as the lead, so the hidden states are audible.

## Data

[POP909](https://github.com/music-x-lab/POP909-Dataset) (MIT license, ISMIR 2020): 909 Chinese pop songs as
MIDI with a separate MELODY track, plus annotated chords, beats and keys.

Facts checked on the full dataset (`python scripts/data_stats.py`):

- 309,423 melody notes, median 339 per song, median pitch 72 (C5)
- beat grid: one row per beat, median 0.75 s (ar. 80 bpm, many slow ballads)
- note onsets fall on quarter beats: 44% on the beat, 35% on the half beat, 21% on the other two 16ths
  -> a 16th note grid keeps every onset, an 8th note grid would lose 1 in 5
- 9.2% of consecutive melody notes overlap, almost all by a small legato amount (median 1/20 of a beat)
  -> the melody is effectively monophonic
- 757 of 909 songs stay in one key; 502 major, 407 minor
- 28 chord qualities; maj and min dominate

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Clone POP909 next to this repo, so the data lives in `../POP909-Dataset/POP909`.

```
pytest
python scripts/data_stats.py
```

## Roadmap

| # | Deliverable | Status |
|---|---|---|
| M1 | Data loading, MIDI reader and writer, representation, Markov chain baselines, held-out likelihood | step 1 done (loading) |
| M2 | HMM from scratch: forward-backward, Viterbi, Baum-Welch, tests against brute force | |
| M3 | Ethereal synth renderer (NumPy to WAV), generation, evaluation and listening | |
