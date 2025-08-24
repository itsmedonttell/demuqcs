# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility for on the fly pitch/tempo change for data augmentation."""

import random
import subprocess as sp
import tempfile
import typing as tp

from . import audio_legacy
import torch
import torchaudio as ta

from .audio import save_audio


class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset: tp.Any, proba: float = 0.2, max_pitch: int = 2, max_tempo: int = 12,
                 tempo_std: int = 5, vocals: tp.List[int] = [3], same: bool = True) -> None:
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.tempo_std = tempo_std
        self.same = same
        self.vocals = vocals

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int((1 - 0.01 * self.max_tempo) * in_length)

        if random.random() < self.proba:
            outs = []
            for idx, stream in enumerate(streams):
                if idx == 0 or not self.same:
                    delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
                    delta_tempo = random.gauss(0, self.tempo_std)
                    delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)
                stream = repitch(
                    stream,
                    delta_pitch,
                    delta_tempo,
                    voice=idx in self.vocals)
                outs.append(stream[:, :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams


def repitch(wav: torch.Tensor, pitch: float, tempo: float, voice: bool = False, quick: bool = False, samplerate: int = 44100) -> torch.Tensor:
    """
    tempo is a relative delta in percentage, so tempo=10 means tempo at 110%!
    pitch is in semi tones.
    Requires `soundstretch` to be installed, see
    https://www.surina.net/soundtouch/soundstretch.html
    """
    infile = tempfile.NamedTemporaryFile(suffix=".wav")
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(wav, infile.name, samplerate, clip='clamp')
    command = [
        "soundstretch",
        infile.name,
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr = ta.load(outfile.name)
    assert sr == samplerate
    return wav
