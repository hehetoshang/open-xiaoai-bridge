"""AEC 单元测试：参考对齐、无播放、双讲、缓冲和格式保护。"""

import unittest

import numpy as np

from core.services.audio.aec import AcousticEchoCanceller


def pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 0.999) * 32768).astype("<i2").tobytes()


def floats(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768


class AcousticEchoCancellerTest(unittest.TestCase):
    def test_disabled_and_no_playback_are_safe(self):
        source = pcm(np.linspace(-0.2, 0.2, 320, dtype=np.float32))
        disabled = AcousticEchoCanceller(enabled=False, delay_ms=0)
        self.assertEqual(source, disabled.process_capture(source))

        enabled = AcousticEchoCanceller(enabled=True, delay_ms=0)
        self.assertEqual(source, enabled.process_capture(source))
        self.assertEqual(1, enabled.snapshot()["underflows"])

    def test_reference_is_resampled_and_buffer_is_bounded(self):
        aec = AcousticEchoCanceller(
            enabled=True, delay_ms=0, max_reference_ms=20
        )
        reference_24k = pcm(np.ones(2400, dtype=np.float32) * 0.1)
        aec.feed_reference(reference_24k, sample_rate=24000)
        snapshot = aec.snapshot()
        self.assertGreater(snapshot["reference_frames"], 0)
        self.assertEqual(1, snapshot["overflows"])
        self.assertLessEqual(snapshot["queued_reference_ms"], 20)

    def test_capture_format_mismatch_is_rejected(self):
        aec = AcousticEchoCanceller(enabled=True, delay_ms=0)
        with self.assertRaisesRegex(ValueError, "format mismatch"):
                aec.process_capture(b"\0\0" * 160, sample_rate=24000)

    def test_non_aligned_capture_frame_is_fully_processed(self):
        aec = AcousticEchoCanceller(enabled=True, delay_ms=0)
        source = pcm(np.linspace(-0.1, 0.1, 720, dtype=np.float32))
        aec.feed_reference(b"\0\0" * 720, sample_rate=16000)
        result = aec.process_capture(source)
        self.assertEqual(len(source), len(result))
        self.assertEqual(source, result)

    def test_adaptive_filter_reduces_aligned_echo(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(0, 0.12, 160 * 160).astype(np.float32)
        # 房间路径：直达 + 2ms 反射；均在自适应滤波器覆盖范围内。
        echo = 0.65 * reference
        echo[32:] += 0.25 * reference[:-32]
        aec = AcousticEchoCanceller(
            enabled=True,
            delay_ms=0,
            filter_ms=10,
            adaptation_rate=0.08,
            max_reference_ms=2000,
        )
        output = []
        for offset in range(0, len(reference), 160):
            ref_block = reference[offset : offset + 160]
            echo_block = echo[offset : offset + 160]
            aec.feed_reference(pcm(ref_block), sample_rate=16000)
            output.append(floats(aec.process_capture(pcm(echo_block))))
        result = np.concatenate(output)
        tail = slice(len(result) // 2, None)
        attenuation = np.sqrt(np.mean(result[tail] ** 2)) / np.sqrt(
            np.mean(echo[tail] ** 2)
        )
        self.assertLess(attenuation, 0.55)
        self.assertGreater(aec.snapshot()["adapted_blocks"], 100)

    def test_double_talk_and_saturation_freeze_adaptation(self):
        aec = AcousticEchoCanceller(enabled=True, delay_ms=0)
        reference = np.ones(160, dtype=np.float32) * 0.01
        near_end = np.ones(160, dtype=np.float32) * 0.2
        aec.feed_reference(pcm(reference), sample_rate=16000)
        aec.process_capture(pcm(near_end))
        self.assertEqual(1, aec.snapshot()["double_talk_blocks"])

        aec.feed_reference(pcm(reference), sample_rate=16000)
        aec.process_capture(pcm(np.ones(160, dtype=np.float32)))
        self.assertEqual(1, aec.snapshot()["saturated_blocks"])


if __name__ == "__main__":
    unittest.main()
