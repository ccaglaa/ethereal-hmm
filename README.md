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

## M2: HMM from scratch (`hmmusic/hmm.py`)

Forward-backward with scaling, Viterbi, posterior decoding, Baum-Welch (EM with optional pseudo counts),
supervised fitting and sampling. Two emission models:
- plain: `B[i, v] = P(note | state)`
- autoregressive (ar1): `B[i, u, v] = P(note | state, previous note)`

Every inference result is tested against brute force over all K^T state paths, and one EM step is
checked against expected counts computed by brute force.

### Why one symbol per note, not per 16th

On 16th frames the HMM with the annotated chords as states scored 2.38 bits per frame, worse than an order 1
Markov chain (2.14): 73% of frames are HOLD or REST, and a chord says nothing about rhythm. So M2 models
the sequence of notes (pitch class only); rhythm gets its own model in M3.

### Results (`python scripts/hmm_experiments.py`, then `python scripts/summarize_hmm.py`)

202,846 / 24,374 / 26,304 notes for train / val / test. Uniform guess: 3.585 bits per note.
Baum-Welch: random Dirichlet start, pseudo count 0.1, stops at a gain below 1e-4 nats per note, 3 seeds (mean +- sd).

| model | test bits per note | chord accuracy % |
|---|---|---|
| Markov order 0 / 1 / 2 / 3 / 4 | 2.679 / 2.404 / **2.358** / 2.366 / 2.473 | |
| always the most common chord (Am) | | 18.1 |
| chord from the current note only | | 28.2 |
| HMM, states = annotated chords, plain | 2.633 | 33.8 |
| HMM, states = annotated chords, ar1 | 2.381 | **36.9** |
| Baum-Welch plain, K = 8 / 25 / 50 | 2.409 / 2.347 / 2.318 | 26.6 / 28.5 / 29.7 |
| Baum-Welch ar1, K = 4 / 8 / 25 / 50 | 2.369 / 2.343 / 2.313 / **2.309** | 21.0 / 22.0 / 23.6 / 23.6 |

Seed sd is 0.003 to 0.017 bits. Chord accuracy for Baum-Welch: each learned state is mapped to the chord it
co-occurs with most on train.

What this shows:

1. **Unsupervised HMMs predict melodies best**, but only a little: 2.31 bits per note vs 2.36 for the best
   Markov chain (about 2%, 3 seed sd).
2. **Chords alone explain melodies poorly.** With the annotated chords as states, the plain HMM (2.63) barely
   beats note frequencies (2.68). Adding the previous note (ar1) brings it to 2.38, close to Markov order 2.
   Melodies move by small steps; the chord mostly decides which of the nearby notes.
3. **Decoding matters.** Viterbi (best single path) gave 22.2% / 24.5% chord accuracy, below the current note
   baseline; posterior decoding (best state per note) gives 33.8% / 36.9%. Viterbi over-smooths because
   chords last several notes (76% of notes keep the previous note's chord).
4. **Baum-Welch does not discover chords.** EM maximizes likelihood, and the hidden variable that helps most
   is not harmony. Plain HMMs with few states learn single notes (K=8: one state per scale note), a Markov
   chain in disguise. Given the previous note (ar1), states learn melodic motion
   (`python scripts/inspect_states.py outputs/hmm_bw_ar1_K8_seed1.npz`): a "repeat" state (88% same note),
   a "rise" state (67% up), "fall" states, a one-note falling step that never stays. Their chord accuracy
   (ar1 ar. 23%) is below the current note baseline.

Consequence for M3: generation uses the HMM with annotated chords as states (ar1). It predicts slightly
worse than the best unsupervised model (2.38 vs 2.31 bits) but its hidden states are chords, which is
exactly what the synth pad needs.

## M3: generation and sound

### Synth (`hmmusic/synth.py`, NumPy only)

- lead: soft additive tone (4 harmonics), two oscillators detuned by +-4 cents, slow vibrato fading in,
  ping-pong delay on a dotted eighth
- pad: triads voiced between E3 and C5 with voice leading (each chord moves as little as possible from the
  previous one), 3 detuned oscillators, 1.6 s attack, 2.5 s release, dark (5 harmonics), plus a soft root bass
- reverb: convolution (FFT) with a synthetic stereo impulse response, decaying noise, RT60 4.5 s
- master: soft clip, fades, peak at -1 dBFS, 16 bit stereo WAV with the standard library
- presets (`PRESETS`), so the same notes can be rendered several ways:
  - `airy` (default): the sound above
  - `dreamy`: softer lead, slower pad, reverb 7 s and wetter, more echoes
  - `glass`: brighter bell like lead plus a quiet copy one octave up (shimmer), lighter pad
  - `pulse`: chord tones as 8th notes (arpeggio) over a quieter pad, shorter reverb, for fast tempos
- the pad attack is the shorter of 1.6 s and 1.5 beats (1 beat for `pulse`): at 120 bpm a fixed 1.6 s attack would still be
  fading in when the next chord arrives

`python scripts/render_song.py` plays a real test song (default: the slowest one, 571) through the synth,
as a reference for the sound.

### Generator (`hmmusic/generate.py`)

An HMM whose hidden chord moves once per beat (chords change on beats, as in the songs) and emits a note at each onset:
- rhythm: P(gap to next onset | position in the beat, previous gap), gaps of 1 to 16 sixteenths
- chords: Witten-Bell Markov chain over the chord at each beat; order 4 chosen on validation songs
  (1.823 / 1.720 / 1.713 / 1.712 bits per beat for orders 1 to 4; higher orders do not fit in memory)
- notes: pitch class from `B[chord, previous pitch class]`, the emission table of the ar1 chord HMM of M2;
  octave = closest to the previous note between G4 and D6

`python scripts/generate.py` writes `outputs/ethereal_seed<k>_AABA.wav` and `.mid` (tracks LEAD and PAD);
`--form=none` for free sampling, or another form such as `--form=ABAB`:
every A is the theme (varied after the first), every other letter is a new contrasting phrase.
Tempo is drawn between 99 and 136 bpm for each seed (`--bpm=99-136`, or a fixed value like `--bpm=110`).
By default each piece is written as `_airy.wav` and `_dreamy.wav` (the two favourites after listening);
`--preset=glass` picks one sound, `--preset=all` writes the same piece once per preset.

### Form (`generate_form`)

Free sampling never repeats anything, while real songs do. `generate_form` arranges 4-bar phrases as AABA:
- A: the theme. Its chords are picked among 50 sampled progressions: at least 3 chord changes (the most
  loopable progressions are otherwise static, staying on a chord is always the likeliest move; real songs
  change ar. 5 times per 4 bars), starting on a home chord (C or Am), then the highest probability of
  going from its last chords back to its first
- A', A'': same rhythm and chords, first 2 bars identical, notes of the last 2 bars sampled again (an answer)
- B: new rhythm and chords, continuing from where A ends
- the last phrase ends with a full close: the last 2 beats of the second to last bar become the dominant
  (G before C, E major before A minor), notes there are sampled again from it, the last one is the scale
  step above the root (D before C, B before A), and the root falls on the downbeat of the last bar, alone
- the final note and chord ring 4 s longer whatever the tempo (`ring_out`); at 130 bpm the last bar
  lasts only ar. 2 s, and endings first felt cut off

### Evaluation (`python scripts/evaluate_generation.py`)

50 generated pieces of 16 bars vs 160 test phrases. The control uses the same rhythm and chords but an
order 2 Markov chain for notes, so the notes ignore the chord. "Repeated" = share of 8-note pitch class
patterns already heard earlier in the same piece (real songs: their first 90 notes, about 16 bars).

| | chord tone | repeat | step 1-2 | leap 3+ | notes per bar | repeated |
|---|---|---|---|---|---|---|
| test songs | 66.4% | 25.0% | 42.1% | 32.8% | 5.2 | 21.2% |
| chord HMM, free | 61.6% | 22.4% | 45.3% | 32.3% | 5.6 | 0.0% |
| Markov 2, no chords (control) | 45.4% | 24.8% | 44.7% | 30.5% | 5.6 | 0.1% |
| chord HMM + AABA form and cadence | **67.0%** | 23.4% | 43.7% | 32.9% | 5.5 | **12.3%** |

- The hidden chords do their job: generated notes are chord tones far more often than notes that ignore
  the chord (62 to 67% vs 45%), close to real melodies (66%). Step sizes match the real melodies.
- Free sampling has no memory beyond a few notes: 0% repeated patterns. Form brings it to 12%, about
  half of real songs (21%), without hurting harmony.
- Not measured: whether it sounds good. That needs listening.

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
python scripts/hmm_experiments.py
python scripts/summarize_hmm.py
python scripts/render_song.py
python scripts/generate.py
python scripts/evaluate_generation.py
```

`hmm_experiments.py` takes ar. 1.5 h. Add `--quick` for a 3 minute check, or split it over two terminals
with `--orders=0` and `--orders=1`.

Sampled melodies are written to `outputs/` as MIDI (open them in GarageBand).

## Roadmap

| # | Deliverable | Status |
|---|---|---|
| M1 | Data loading, MIDI reader and writer, representation, Markov chain baselines, held-out likelihood | Done |
| M2 | HMM from scratch: forward-backward, Viterbi, Baum-Welch, tests against brute force | Done |
| M3 | Ethereal synth renderer (NumPy to WAV), generation, evaluation and listening | Done |
