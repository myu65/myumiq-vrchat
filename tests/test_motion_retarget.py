import json

import numpy as np
import pytest
from test_articulated_body import fixture

from myumiq_vrchat.motion import rotation
from myumiq_vrchat.motion_retarget import RetargetProfile
from myumiq_vrchat.whole_body import vector


def test_transfer_preserves_destination_lengths_and_explicit_reference():
    rig, states = fixture()
    profile = RetargetProfile.model_validate_json(
        json.dumps(
            {
                "rig": rig.model_dump(mode="json"),
                "reference_root": states[0].root.tolist(),
                "reference_rotations": states[0].rotations.tolist(),
                "source_nodes": {name: i for i, name in enumerate(rig.names)},
                "source_reference_clip": "rest",
                "source_root": 0,
                "basis": np.eye(3).tolist(),
                "translation_scale": 0.5,
            }
        )
    )

    class Motion:
        nodes = [{}] * len(rig.names)

        def duration(self, clip):
            return 0.3

        def world_matrices(self, clip, time_s):
            result = {i: np.eye(4) for i in range(len(self.nodes))}
            if clip != "rest":
                for value in result.values():
                    value[:3, :3] = rotation((0.0, 0.0, np.sin(time_s / 2), np.cos(time_s / 2)))
                    value[:3, 3] = [2 * time_s, 0, 0]
            return result

    frames = profile.retarget(Motion(), "turn", hz=10)
    assert frames[-1][0] == 0.3
    np.testing.assert_allclose(vector(frames[0][1]), vector(rig.forward(states[0])), atol=1e-8)
    a, b = vector(frames[0][1])[:, :3], vector(frames[-1][1])[:, :3]
    np.testing.assert_allclose(
        np.linalg.norm(a[:, None] - a[None, :], axis=2),
        np.linalg.norm(b[:, None] - b[None, :], axis=2),
        atol=1e-8,
    )
    assert frames[-1][1].pelvis.position[0] == pytest.approx(0.3)
    invalid = json.loads(profile.model_dump_json())
    invalid["source_nodes"].pop("joint-0")
    with pytest.raises(ValueError, match="mapping"):
        RetargetProfile.model_validate_json(json.dumps(invalid))
