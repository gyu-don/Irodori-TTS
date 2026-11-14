import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import torch

from irodori_tts.inference_runtime import (
    _load_audio,
    _move_inference_module,
    list_available_runtime_precisions,
    save_wav,
)


class AudioFallbackTests(unittest.TestCase):
    def test_precision_conversion_preserves_complex_rope_and_integer_buffers(self):
        module = torch.nn.Linear(2, 2)
        module.register_buffer("rope", torch.tensor([1 + 2j], dtype=torch.complex64))
        module.register_buffer("indices", torch.tensor([3], dtype=torch.int64))
        module.register_buffer("scale", torch.tensor([0.5]))
        before = module.weight.detach().clone()
        _move_inference_module(module, device=torch.device("cpu"), dtype=torch.bfloat16)
        torch.testing.assert_close(module.weight, before.bfloat16())
        self.assertEqual(module.scale.dtype, torch.bfloat16)
        self.assertEqual(module.rope.dtype, torch.complex64)
        self.assertEqual(module.rope.item(), 1 + 2j)
        self.assertEqual(module.indices.dtype, torch.int64)

    def test_precision_default_follows_selected_device(self):
        self.assertEqual(list_available_runtime_precisions("cpu"), ["fp32"])
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_capability", return_value=(12, 1)):
                self.assertEqual(list_available_runtime_precisions("cuda")[0], "bf16")
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                self.assertEqual(list_available_runtime_precisions("cuda")[0], "fp32")

    def test_bfloat16_mono_and_stereo_round_trip_without_torchcodec(self):
        # ARM64 wheels lack TorchCodec. Both torchaudio operations raise ImportError.
        with tempfile.TemporaryDirectory() as directory:
            for channels in (1, 2):
                with self.subTest(channels=channels):
                    audio = torch.linspace(-0.5, 0.5, 200).repeat(channels, 1).bfloat16()
                    path = Path(directory) / f"{channels}.wav"
                    with patch("torchaudio.save", side_effect=ImportError("torchcodec")):
                        save_wav(path, audio, 48000)
                    with patch("torchaudio.load", side_effect=ImportError("torchcodec")):
                        loaded, sample_rate = _load_audio(path)
                    self.assertEqual(sample_rate, 48000)
                    self.assertEqual(loaded.shape, audio.shape)
                    torch.testing.assert_close(loaded, audio.float(), atol=1 / 32768, rtol=0)

    def test_vector_waveform_and_runtime_backend_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            with patch("torchaudio.save", side_effect=RuntimeError("backend unavailable")):
                save_wav(path, torch.ones(2400) * 0.25, 24000)
            data, sample_rate = sf.read(path)
            self.assertEqual(data.shape, (2400,))
            self.assertEqual(sample_rate, 24000)


if __name__ == "__main__":
    unittest.main()
