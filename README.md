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

## Representation (`hmmusic/frames.py`)

- Every song is transposed to C major or A minor (both use only the white keys); 152 songs that change key are skipped -> 757 songs
- One frame per 16th note. Melody symbol per frame: 0..11 = a new note with that pitch class, 12 = HOLD, 13 = REST.
  Octaves are dropped and chosen again when decoding (closest to the previous note, range C4 to C6)
- Chords reduced to root + major-like or minor-like (24) + no chord. Most common: Am, G, C, F, Dm, Em
- Quantization loses 508 of 254,032 notes (0.2%, notes landing on the same 16th)
- Songs are cut into phrases at silences of 2 bars or more (intros, interludes); otherwise 20% of frames are long silences
- Split by song, seed 0: 605 / 76 / 76 songs for train / val / test

The melodies are close to pentatonic: C D E G A make 24% of frames, F and B only 2.6%. 64% of note onsets are a
tone of the current chord triad, which is the signal the HMM will try to discover without labels.

## M1 results: Markov chain baselines (`python scripts/markov_baselines.py`)

Held-out bits per frame on test songs (lower is better; a uniform guess over 14 symbols is 3.807 bits).

| order | additive (alpha tuned on val) | Witten-Bell |
|---|---|---|
| 0 | 2.427 | |
| 1 | 2.136 | 2.136 |
| 2 | 1.920 | 1.922 |
| 3 | 1.891 | 1.899 |
| 4 | **1.848** | 1.866 |
| 5 | | 1.974 |

Likelihood and sample quality disagree. The additive order 4 model scores best, but its samples are chromatic noise
(`C F B F# - Ab B D E ...`): most of the 15^4 contexts were never seen, add-alpha makes their next symbol uniform,
and once a sample leaves the scale it never comes back. Test melodies rarely visit unseen contexts, so the score
does not show it. Witten-Bell backs off to shorter contexts and its samples stay in the scale (`E - D - E - A ...`).
Lesson: report likelihood and listen.

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
python scripts/markov_baselines.py
```

Sampled melodies are written to `outputs/` as MIDI (open them in GarageBand).

## Roadmap

| # | Deliverable | Status |
|---|---|---|
| M1 | Data loading, MIDI reader and writer, representation, Markov chain baselines, held-out likelihood | Done |
| M2 | HMM from scratch: forward-backward, Viterbi, Baum-Welch, tests against brute force | |
| M3 | Ethereal synth renderer (NumPy to WAV), generation, evaluation and listening | |
