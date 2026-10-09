import numpy as np
import pytest

from hmmusic.decode import nearest_pitch, obs_to_notes
from hmmusic.frames import HOLD, REST


@pytest.mark.parametrize("pc, previous, pitch", [(0, 72, 72), (11, 72, 71), (7, 72, 67), (4, 60, 64), (0, 90, 84), (11, 50, 71)])
def test_nearest_pitch(pc, previous, pitch):
    assert nearest_pitch(pc, previous) == pitch


def test_obs_to_notes():
    # 80 bpm -> one 16th frame = 0.1875 s
    obs = np.array([0, HOLD, 2, REST, HOLD, 11])
    notes = obs_to_notes(obs, bpm=80, start_pitch=72)
    frame = 0.1875
    assert [(n.pitch, n.start, n.end) for n in notes] == [
        (72, 0.0, 2 * frame),
        (74, 2 * frame, 3 * frame),
        (71, 5 * frame, 6 * frame),
    ]


def test_obs_to_notes_all_rest():
    assert obs_to_notes(np.array([REST, HOLD, REST])) == []
