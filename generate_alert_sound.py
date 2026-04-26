import os
import struct
import math

def write_wav(filename, samples, sample_rate=44100, num_channels=1, bit_depth=16):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    num_samples = len(samples)
    byte_rate    = sample_rate * num_channels * (bit_depth // 8)
    block_align  = num_channels * (bit_depth // 8)
    data_size    = num_samples * block_align

    with open(filename, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels,
                            sample_rate, byte_rate, block_align, bit_depth))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            val = int(clamped * 32767)
            f.write(struct.pack("<h", val))

def generate_beep(freq, duration, sample_rate=44100, amplitude=0.5):
    n = int(sample_rate * duration)
    return [amplitude * math.sin(2 * math.pi * freq * i / sample_rate)
            for i in range(n)]

if __name__ == "__main__":
    sr = 44100
    # Two beeps: 880 Hz (A5) then 1046 Hz (C6)
    beep1   = generate_beep(880,  0.18, sr)
    silence = [0.0] * int(sr * 0.05)
    beep2   = generate_beep(1046, 0.18, sr)

    samples = beep1 + silence + beep2
    out = "assets/alert.wav"
    write_wav(out, samples, sr)
    print(f"Saved: {out}")
