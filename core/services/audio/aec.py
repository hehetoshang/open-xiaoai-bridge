"""基于实际扬声器 PCM 参考信号的轻量频域自适应回声消除。"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import resample_poly


@dataclass
class AECDiagnostics:
    capture_frames: int = 0
    reference_frames: int = 0
    underflows: int = 0
    overflows: int = 0
    double_talk_blocks: int = 0
    saturated_blocks: int = 0
    adapted_blocks: int = 0


class AcousticEchoCanceller:
    """分块频域 NLMS AEC。

    参考信号由播放器在送往扬声器前写入，采集信号以 10ms 分块处理。
    `delay_ms` 用于补偿播放缓冲和声学路径的固定延迟；滤波器继续学习房间
    多径响应。双讲和削波时冻结自适应，避免把近端人声学习进回声模型。
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        capture_rate: int = 16000,
        block_ms: int = 10,
        delay_ms: int = 120,
        filter_ms: int = 10,
        adaptation_rate: float = 0.12,
        max_reference_ms: int = 3000,
    ):
        self._lock = threading.RLock()
        self.enabled = enabled
        self.capture_rate = capture_rate
        self.block_size = max(80, int(capture_rate * block_ms / 1000))
        self.delay_samples = max(0, int(capture_rate * delay_ms / 1000))
        self.filter_samples = max(32, int(capture_rate * filter_ms / 1000))
        self.adaptation_rate = max(0.01, min(float(adaptation_rate), 0.5))
        self.max_reference_samples = max(
            self.block_size * 2, int(capture_rate * max_reference_ms / 1000)
        )
        self._reference = np.zeros(self.delay_samples, dtype=np.float32)
        self._previous_reference = np.zeros(self.block_size, dtype=np.float32)
        self._fft_size = 1 << (2 * self.block_size - 1).bit_length()
        self._filter = np.zeros(self._fft_size, dtype=np.complex64)
        self._power = np.ones(self._fft_size, dtype=np.float32) * 1e-3
        self.diagnostics = AECDiagnostics()

    def configure(self, config: dict | None) -> None:
        config = config or {}
        with self._lock:
            enabled = bool(config.get("enabled", False))
            changed = (
                int(config.get("delay_ms", 120)) != self.delay_samples * 1000 // self.capture_rate
                or float(config.get("adaptation_rate", 0.12)) != self.adaptation_rate
            )
            self.enabled = enabled
            self.adaptation_rate = max(
                0.01, min(float(config.get("adaptation_rate", 0.12)), 0.5)
            )
            self.delay_samples = max(
                0, int(self.capture_rate * int(config.get("delay_ms", 120)) / 1000)
            )
            if changed:
                self.reset()

    def reset(self) -> None:
        with self._lock:
            self._reference = np.zeros(self.delay_samples, dtype=np.float32)
            self._previous_reference.fill(0)
            self._filter.fill(0)
            self._power.fill(1e-3)

    def feed_reference(
        self,
        pcm: bytes,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        if not self.enabled or not pcm:
            return
        samples = self._decode_pcm(pcm, channels)
        if sample_rate != self.capture_rate:
            divisor = int(np.gcd(sample_rate, self.capture_rate))
            samples = resample_poly(
                samples,
                self.capture_rate // divisor,
                sample_rate // divisor,
            ).astype(np.float32, copy=False)
        with self._lock:
            self._reference = np.concatenate((self._reference, samples))
            self.diagnostics.reference_frames += len(samples)
            if len(self._reference) > self.max_reference_samples:
                dropped = len(self._reference) - self.max_reference_samples
                self._reference = self._reference[dropped:]
                self.diagnostics.overflows += 1

    def process_capture(
        self,
        pcm: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> bytes:
        if not self.enabled or not pcm:
            return pcm
        if sample_rate != self.capture_rate:
            raise ValueError(
                f"AEC capture format mismatch: expected {self.capture_rate}Hz, got {sample_rate}Hz"
            )
        capture = self._decode_pcm(pcm, channels)
        with self._lock:
            needed = len(capture)
            available = min(needed, len(self._reference))
            reference = np.zeros(needed, dtype=np.float32)
            if available:
                reference[:available] = self._reference[:available]
                self._reference = self._reference[available:]
            if available < needed:
                self.diagnostics.underflows += 1

            output = np.empty_like(capture)
            cursor = 0
            while cursor + self.block_size <= needed:
                end = cursor + self.block_size
                output[cursor:end] = self._process_block(
                    capture[cursor:end], reference[cursor:end]
                )
                cursor = end
            if cursor < needed:
                # 回调帧长不一定是 10ms 的整数倍；补零处理尾块，但只返回真实样本。
                remaining = needed - cursor
                capture_tail = np.zeros(self.block_size, dtype=np.float32)
                reference_tail = np.zeros(self.block_size, dtype=np.float32)
                capture_tail[:remaining] = capture[cursor:]
                reference_tail[:remaining] = reference[cursor:]
                output[cursor:] = self._process_block(
                    capture_tail, reference_tail
                )[:remaining]
            self.diagnostics.capture_frames += needed

        return self._encode_pcm(output)

    def snapshot(self) -> dict:
        with self._lock:
            data = asdict(self.diagnostics)
            data.update(
                {
                    "enabled": self.enabled,
                    "delay_ms": self.delay_samples * 1000 // self.capture_rate,
                    "queued_reference_ms": len(self._reference) * 1000 // self.capture_rate,
                }
            )
            return data

    def _process_block(self, capture: np.ndarray, reference: np.ndarray) -> np.ndarray:
        joined = np.concatenate((self._previous_reference, reference))
        spectrum = np.fft.rfft(joined, n=self._fft_size)
        estimated_full = np.fft.irfft(
            self._filter[: len(spectrum)] * spectrum,
            n=self._fft_size,
        )
        estimated = estimated_full[
            self.block_size : self.block_size * 2
        ].astype(np.float32, copy=False)
        if not np.all(np.isfinite(estimated)):
            self._filter.fill(0)
            self._power.fill(1e-3)
            estimated = np.zeros_like(capture)
        error = capture - estimated

        ref_rms = float(np.sqrt(np.mean(reference * reference) + 1e-12))
        capture_rms = float(np.sqrt(np.mean(capture * capture) + 1e-12))
        estimated_rms = float(np.sqrt(np.mean(estimated * estimated) + 1e-12))
        if estimated_rms > max(0.5, capture_rms * 4):
            # 数值保护：异常模型不能向 KWS 注入比采集信号更强的伪影。
            self._filter.fill(0)
            self._power.fill(1e-3)
            estimated.fill(0)
            error = capture.copy()
        saturated = bool(np.max(np.abs(capture)) >= 0.995)
        double_talk = ref_rms > 1e-4 and capture_rms > max(0.08, ref_rms * 2.5)

        if saturated:
            self.diagnostics.saturated_blocks += 1
        if double_talk:
            self.diagnostics.double_talk_blocks += 1
        if ref_rms > 1e-4 and not saturated and not double_talk:
            padded_error = np.concatenate(
                (np.zeros(self.block_size, dtype=np.float32), error)
            )
            error_spectrum = np.fft.rfft(padded_error, n=self._fft_size)
            power = np.abs(spectrum) ** 2
            bins = len(spectrum)
            self._power[:bins] = 0.8 * self._power[:bins] + 0.2 * power
            updated = self._filter[:bins] + (
                self.adaptation_rate
                * np.conj(spectrum)
                * error_spectrum
                / (self._power[:bins] + 1e-5)
            )
            impulse = np.fft.irfft(updated, n=self._fft_size)
            impulse[self.filter_samples :] = 0
            norm = float(np.linalg.norm(impulse[: self.filter_samples]))
            if not np.isfinite(norm):
                impulse.fill(0)
            elif norm > 2.0:
                impulse *= 2.0 / norm
            self._filter[:bins] = np.fft.rfft(impulse, n=self._fft_size)
            self.diagnostics.adapted_blocks += 1

        self._previous_reference = reference.copy()
        return np.clip(error, -1.0, 0.9999695)

    @staticmethod
    def _decode_pcm(pcm: bytes, channels: int) -> np.ndarray:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        raw = np.frombuffer(pcm, dtype="<i2")
        if channels > 1:
            if len(raw) % channels:
                raise ValueError("PCM frame is not aligned to channel count")
            raw = raw.reshape(-1, channels).mean(axis=1)
        return raw.astype(np.float32) / 32768.0

    @staticmethod
    def _encode_pcm(samples: np.ndarray) -> bytes:
        return (np.clip(samples, -1.0, 0.9999695) * 32768.0).astype("<i2").tobytes()


AEC = AcousticEchoCanceller()
