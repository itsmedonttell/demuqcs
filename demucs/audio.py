# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import subprocess as sp
from pathlib import Path

import lameenc
import julius
import numpy as np
from . import audio_legacy
import torch
import torchaudio as ta
import typing as tp

from .utils import temp_filenames


def _read_info(path: Path) -> dict:
    stdout_data = sp.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
    return json.loads(stdout_data.decode('utf-8'))


class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info: tp.Optional[dict] = None

    def __repr__(self) -> str:
        features: tp.List[tp.Tuple[str, tp.Any]] = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self) -> dict:
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self) -> float:
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self) -> tp.List[int]:
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self) -> int:
        return len(self._audio_streams)

    def channels(self, stream: int = 0) -> int:
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream: int = 0) -> int:
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time: tp.Optional[float] = None,
             duration: tp.Optional[float] = None,
             streams: tp.Union[slice, int, tp.List[int]] = slice(None),
             samplerate: tp.Optional[int] = None,
             channels: tp.Optional[int] = None) -> torch.Tensor:
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
        """
        streams_array = np.array(range(len(self)))[streams]
        single = not isinstance(streams_array, np.ndarray)
        if single:
            streams_list = [streams_array]
        else:
            streams_list = streams_array.tolist()

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        with temp_filenames(len(streams_list)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams_list, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav_numpy = np.fromfile(filename, dtype=np.float32)
                wav_tensor = torch.from_numpy(wav_numpy)
                wav_tensor = wav_tensor.view(-1, self.channels()).t()
                if channels is not None:
                    wav_tensor = convert_audio_channels(wav_tensor, channels)
                if target_size is not None:
                    wav_tensor = wav_tensor[..., :target_size]
                wavs.append(wav_tensor)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_samplerate: int, to_samplerate: int, channels: int) -> torch.Tensor:
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to 16 bits integer PCM format."""
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    else:
        return wav.float() / (2**15 - 1)


def as_dtype_pcm(wav: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert audio to either f32 pcm or i16 pcm depending on the given dtype."""
    if wav.dtype.is_floating_point:
        return f32_pcm(wav)
    else:
        return i16_pcm(wav)


def encode_mp3(wav: torch.Tensor, path: tp.Union[str, Path], samplerate: int = 44100, bitrate: int = 320, quality: int = 2, verbose: bool = False) -> None:
    """Save given audio as mp3. This should work on all OSes."""
    C, T = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav_cpu = wav.data.cpu()
    wav_numpy = wav_cpu.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav_numpy.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def prevent_clip(wav: torch.Tensor, mode: tp.Optional[str] = 'rescale') -> torch.Tensor:
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max().item(), 1.0)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav: torch.Tensor,
               path: tp.Union[str, Path],
               samplerate: int,
               bitrate: int = 320,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False,
               preset: tp.Literal[2, 3, 4, 5, 6, 7] = 2):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`. Use `preset` to set mp3 quality:
    2 for highest quality, 7 for fastest speed
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    elif suffix == ".flac":
        ta.save(str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")
