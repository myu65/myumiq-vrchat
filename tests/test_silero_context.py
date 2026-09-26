import numpy as np

from myumiq_vrchat.audio import AudioFrame, SileroVAD


def test_silero_passes_previous_frame_context_to_onnx():
    class Session:
        inputs = []

        def run(self, _, feed):
            self.inputs.append(feed["input"].copy())
            return np.array([[0.0]]), feed["state"]

    vad = SileroVAD.__new__(SileroVAD)
    vad._numpy = np
    vad._session = Session()
    vad._state = np.zeros((2, 1, 128), dtype=np.float32)
    vad._context = np.zeros((1, 64), dtype=np.float32)
    vad.threshold, vad.release_frames = 0.5, 4
    vad.active, vad.silent, vad.buffer = False, 0, []
    vad.accept(AudioFrame((0.25,) * 512, 16000, 1.0))
    vad.accept(AudioFrame((0.5,) * 512, 16000, 2.0))
    first, second = vad._session.inputs
    assert first.shape == second.shape == (1, 576)
    np.testing.assert_array_equal(first[:, :64], 0)
    np.testing.assert_array_equal(second[:, :64], 0.25)
    np.testing.assert_array_equal(second[:, 64:], 0.5)
