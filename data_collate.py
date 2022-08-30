import os.path
import random
import numpy as np
import torch
import re
import torch.utils.data

# import commons
# from glow.utils import load_wav_to_torch, load_filepaths_and_text
# from text import text_to_sequence, cmudict, custom_text_to_sequence
# from text.symbols import symbols
import json

import kaldiio
from tqdm import tqdm


class BaseCollate:
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def collate_text_mel(self, batch: [dict]):
        """
        :param batch: list of dicts
        """
        utt = list(map(lambda x: x['utt'], batch))
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phn_ids']) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['phn_ids']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        dur_padded = torch.LongTensor(len(batch), max_input_len)
        dur_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]]['dur']
            dur_padded[i, :dur.size(0)] = dur

        utt_name = np.array(utt)[ids_sorted_decreasing].tolist()
        if isinstance(utt_name, str):
            utt_name = [utt_name]

        res = {
            "utt": utt_name,
            "text_padded": text_padded,
            "input_lengths": input_lengths,
            "mel_padded": mel_padded,
            "output_lengths": output_lengths,
            "dur_padded": dur_padded
        }
        return res, ids_sorted_decreasing


class SpkIDCollate(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        spk_ids = torch.LongTensor(list(map(lambda x: x["spk_ids"], batch)))
        spk_ids = spk_ids[ids_sorted_decreasing]
        base_data.update({
            "spk_ids": spk_ids
        })
        return base_data


class XvectorCollate(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        xvectors = torch.cat(list(map(lambda x: x["xvector"].unsqueeze(0), batch)), dim=0)
        xvectors = xvectors[ids_sorted_decreasing]
        base_data.update({
            "xvector": xvectors
        })
        return base_data


class SpkIDCollateWithPE(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        spk_ids = torch.LongTensor(list(map(lambda x: x["spk_ids"], batch)))
        spk_ids = spk_ids[ids_sorted_decreasing]

        # num_var = batch[0]["var"].size(0)
        max_target_len = max([x["pitch"].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        pitch_padded = torch.FloatTensor(len(batch), max_target_len)
        pitch_padded.zero_()
        energy_padded = torch.FloatTensor(len(batch), max_target_len)
        energy_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]]["pitch"]
            pitch_padded[i, :, :pitch.size(1)] = pitch
            energy = batch[ids_sorted_decreasing[i]]['energy']
            energy_padded[i, :, :pitch.size(1)] = energy

        base_data.update({
            "spk_ids": spk_ids,
            "pitch_padded": pitch_padded,
            "energy_padded": energy_padded
        })
        return base_data


class XvectorCollateWithPE(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        xvectors = torch.cat(list(map(lambda x: x["xvector"].unsqueeze(0), batch)), dim=0)
        xvectors = xvectors[ids_sorted_decreasing]

        max_target_len = max([x["pitch"].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        pitch_padded = torch.FloatTensor(len(batch), max_target_len)
        pitch_padded.zero_()
        energy_padded = torch.FloatTensor(len(batch), max_target_len)
        energy_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]]["pitch"]
            pitch_padded[i, :, :pitch.size(1)] = pitch
            energy = batch[ids_sorted_decreasing[i]]['energy']
            energy_padded[i, :, :pitch.size(1)] = energy

        base_data.update({
            "xvector": xvectors,
            "pitch_padded": pitch_padded,
            "energy_padded": energy_padded
        })
        return base_data
