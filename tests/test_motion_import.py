import json
import math
import struct

import pytest

np = pytest.importorskip("numpy")

from myumiq_vrchat.motion import GltfMotion  # noqa: E402


def source(tmp_path, times=(0.0, 1.0)):
    arrays = [np.array(times, dtype="<f4"), np.array([[0, 0, 0], [2, 0, 0]], dtype="<f4")]
    payload = b"".join(a.tobytes() for a in arrays)
    (tmp_path / "motion.bin").write_bytes(payload)
    data = {
        "buffers": [{"uri": "motion.bin", "byteLength": len(payload)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 8},
            {"buffer": 0, "byteOffset": 8, "byteLength": 24},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "type": "SCALAR", "count": 2},
            {"bufferView": 1, "componentType": 5126, "type": "VEC3", "count": 2},
        ],
        "nodes": [
            {"name": "parent", "children": [1], "rotation": [0, math.sqrt(0.5), 0, math.sqrt(0.5)]},
            {"name": "child", "translation": [1, 0, 0]},
        ],
        "animations": [
            {
                "name": "move",
                "samplers": [{"input": 0, "output": 1}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            }
        ],
    }
    path = tmp_path / "motion.gltf"
    path.write_text(json.dumps(data))
    return path, data


def test_parent_rotation_and_interpolated_translation_compose(tmp_path):
    path, _ = source(tmp_path)
    motion = GltfMotion(path)
    child = motion.world_matrices("move", 0.5)[1]
    assert child[:3, 3] == pytest.approx([1, 0, -1])
    assert np.linalg.det(child[:3, :3]) == pytest.approx(1)


def test_duplicate_timestamps_fail_instead_of_dividing_by_zero(tmp_path):
    path, _ = source(tmp_path, times=(0.0, 0.0))
    with pytest.raises(ValueError, match="increasing"):
        GltfMotion(path).world_matrices("move", 0.0)


def test_external_buffer_escape_is_rejected(tmp_path):
    path, data = source(tmp_path)
    data["buffers"][0]["uri"] = "../outside.bin"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="local dataset"):
        GltfMotion(path)


def test_glb_embedded_buffer_matches_gltf_and_rejects_truncation(tmp_path):
    path, data = source(tmp_path)
    expected = GltfMotion(path).world_matrices("move", 0.5)[1]
    del data["buffers"][0]["uri"]
    encoded = json.dumps(data).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = (tmp_path / "motion.bin").read_bytes()
    chunks = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    payload = b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks
    glb = tmp_path / "motion.glb"
    glb.write_bytes(payload)
    np.testing.assert_allclose(GltfMotion(glb).world_matrices("move", 0.5)[1], expected)
    assert GltfMotion(glb).source_files == [glb]
    glb.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="GLB"):
        GltfMotion(glb)
