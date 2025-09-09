import random
import torch
import torchaudio

class SpaceCommsAugment:
    """Augment audio to emulate space-to-ground voice constraints."""
    def __init__(self, sr=16000, cfg=None):
        self.sr = sr
        self.cfg = cfg or {}
        self.resample_to = int(self.cfg.get("resample_to_hz", 8000))
        self._to8k = torchaudio.transforms.Resample(sr, self.resample_to)
        self._to16k = torchaudio.transforms.Resample(self.resample_to, sr)

    def _bandlimit(self, wav, sr):
        lo, hi = self.cfg.get("bandpass_hz", [300, 3400])
        return torchaudio.functional.bandpass_biquad(wav, sr, lo, Q=0.707, highcut=hi)

    def _add_noise(self, wav):
        snr_min, snr_max = self.cfg.get("snr_db_range", [0, 15])
        snr = random.uniform(snr_min, snr_max)
        sig_pow = wav.pow(2).mean().clamp(min=1e-9)
        noise_pow = sig_pow / (10 ** (snr / 10))
        noise = torch.randn_like(wav) * torch.sqrt(noise_pow)
        return wav + noise

    def _packet_loss(self, wav):
        # apply at 8k then upsample back
        wav8 = self._to8k(wav)
        sr8 = self.resample_to
        dur_ms = self.cfg.get("packet_loss_ms", [30, 120])
        n = wav8.shape[-1]
        # 1–5 bursts per ~10s
        bursts = max(1, int(n / (sr8 * 2)))
        bursts = random.randint(1, bursts)
        for _ in range(bursts):
            ms = random.randint(*dur_ms)
            L = int(sr8 * ms / 1000)
            i = random.randint(0, max(0, n - L - 1))
            wav8[:, i:i+L] = 0.0
        return self._to16k(wav8)

    def __call__(self, wav):
        if not self.cfg.get("enable", True):
            return wav

        # narrowband hop + bandlimit
        if random.random() < self.cfg.get("p_bandlimit", 0.9):
            wav = self._to8k(wav)
            wav = self._bandlimit(wav, self.resample_to)
            wav = self._to16k(wav)

        if random.random() < self.cfg.get("p_noise", 0.7):
            wav = self._add_noise(wav)

        if random.random() < self.cfg.get("p_packet_loss", 0.4):
            wav = self._packet_loss(wav)

        # peak normalize
        peak = wav.abs().max().clamp(min=1e-6)
        return (wav / peak) * 0.98
