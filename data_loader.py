import os.path
import random
import numpy as np
import torch
import re
import torch.utils.data

import json

import kaldiio
from tqdm import tqdm


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, utts: str, hparams, feats_scp: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str):
        """
        :param utts: file path. A list of utts for this loader. These are the only utts that this loader has access.
        This loader only deals with text, duration and feats. Other files despite `utts` can be larger.
        """
        self.n_mel_channels = hparams.audio.mel.n_mel_channels
        self.sampling_rate = hparams.audio.sampling_rate
        self.utts = self.get_utts(utts)
        self.utt2phn, self.phn2id = self.get_utt2phn(utt2phns, phn2id)
        self.vocab_len = len(self.phn2id.keys())
        self.utt2phn_dur = self.get_utt2phn_dur(utt2phn_duration)
        self.utt2feat = self.get_utt2feat(feats_scp)

        self.add_blank = getattr(hparams, "add_blank", False)

    def get_utts(self, utts: str) -> list:
        with open(utts, 'r') as f:
            L = f.readlines()
            L = list(map(lambda x: x.strip(), L))
            random.seed(1234)
            random.shuffle(L)
        return L

    def get_utt2phn(self, utt2phns: str, phn2id: str) -> (dict, dict):
        res = dict()
        with open(utt2phns, 'r') as f:
            for l in f.readlines():
                res[l.split()[0]] = l.strip().split()[1:]

        res_phn2id = dict()
        with open(phn2id, 'r') as f:
            for l in f.readlines():
                res_phn2id[l.split()[0]] = int(l.strip().split()[1])

        return res, res_phn2id

    def get_utt2phn_dur(self, utt2phn_duration: str) -> dict:
        res = dict()
        with open(utt2phn_duration, 'r') as f:
            for l in f.readlines():
                uttid = l.split()[0]
                # map to integer
                durs = list(map(int, l.strip().split()[1:]))
                res[uttid] = durs
        return res

    def get_utt2feat(self, feats_scp: str):
        utt2feat = kaldiio.load_scp(feats_scp)  # lazy load mode
        print(f"Succeed reading feats from {feats_scp}")
        return utt2feat

    def get_mel_from_kaldi(self, utt):
        feat = self.utt2feat[utt]
        feat = torch.FloatTensor(feat).squeeze()
        assert self.n_mel_channels in feat.shape
        if feat.shape[0] == self.n_mel_channels:
            return feat
        else:
            return feat.T

    def get_text(self, utt):
        phn_seq = self.utt2phn[utt]
        phn_id_seq = list(map(lambda x: self.phn2id[x], phn_seq))
        if self.add_blank:
            phn_id_seq = intersperse(phn_id_seq, max(self.phn2id.values()) + 1)

        return torch.LongTensor(phn_id_seq)

    def get_dur_from_kaldi(self, utt):
        return torch.LongTensor(self.utt2phn_dur[utt])

    def __getitem__(self, index):
        res = self.get_mel_text_pair(self.utts[index])
        return res

    def __len__(self):
        return len(self.utts)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class SpkIDLoader(BaseLoader):
    def __init__(self, utts: str, hparams, feats_scp: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk: str):
        """
        :param utt2spk: json file path (utt name -> spk id)
        This loader loads speaker as a speaker ID for embedding table
        """
        super(SpkIDLoader, self).__init__(utts, hparams, feats_scp, utt2phns, phn2id, utt2phn_duration)
        self.utt2spk = self.get_utt2spk(utt2spk)

    def get_utt2spk(self, utt2spk: str) -> dict:
        with open(utt2spk, 'r') as f:
            res = json.load(f)
        return res

    def get_mel_text_pair(self, utt):
        # separate filename and text
        spkid = self.utt2spk[utt]
        phn_ids = self.get_text(utt)
        mel = self.get_mel_from_kaldi(utt)
        dur = self.get_dur_from_kaldi(utt)

        assert sum(dur) == mel.shape[1], f"Frame length mismatch: utt {utt}, dur: {sum(dur)}, mel: {mel.shape[1]}"
        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "dur": dur,
            "spk_ids": spkid
        }
        return res

    def __getitem__(self, index):
        res = self.get_mel_text_pair(self.utts[index])
        return res

    def __len__(self):
        return len(self.utts)


class SpkIDLoaderWithPE(SpkIDLoader):
    def __init__(self, utts: str, hparams, feats_scp: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk: str, var_scp: str):
        """
        This loader loads speaker ID together with variance (4-dim pitch, 1-dim energy)
        """
        super(SpkIDLoaderWithPE, self).__init__(utts, hparams, feats_scp, utt2phns, phn2id, utt2phn_duration, utt2spk)
        self.utt2var = self.get_utt2var(var_scp)

    def get_utt2var(self, utt2var: str) -> dict:
        res = kaldiio.load_scp(utt2var)
        print(f"Succeed reading feats from {utt2var}")
        return res

    def get_var_from_kaldi(self, utt):
        # returns raw log pitch and energy
        # var comprises pov, pitch feature, delta pitch feature, and raw log pitch, and energy. See notion.
        var = self.utt2var[utt]
        var = torch.FloatTensor(var).squeeze()
        assert 5 in var.shape
        if var.shape[0] == 5:
            return var[3, :], var[4, :]
        else:
            return var.T[3, :], var.T[4, :]

    def get_mel_text_pair(self, utt):
        # separate filename and text
        spkid = self.utt2spk[utt]
        phn_ids = self.get_text(utt)
        mel = self.get_mel_from_kaldi(utt)
        dur = self.get_dur_from_kaldi(utt)
        pitch, energy = self.get_var_from_kaldi(utt)

        assert sum(dur) == mel.shape[1] == pitch.shape[0] == energy.shape[0], \
            f"Frame length mismatch: utt {utt}, dur: {sum(dur)}, mel: {mel.shape[1]}, pitch: {pitch.shape[1]}"

        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "dur": dur,
            "spk_ids": spkid,
            "pitch": pitch,
            "energy": energy
        }
        return res


class XvectorLoader(BaseLoader):
    def __init__(self, utts: str, hparams, feats_scp: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk_name: str, spk_xvector_scp: str):
        """
        :param utt2spk_name: like kaldi-style utt2spk
        :param spk_xvector_scp: kaldi-style speaker-level xvector.scp
        """
        super(XvectorLoader, self).__init__(utts, hparams, feats_scp, utt2phns, phn2id, utt2phn_duration)
        self.utt2spk = self.get_utt2spk(utt2spk_name)
        self.spk2xvector = self.get_spk2xvector(spk_xvector_scp)

    def get_utt2spk(self, utt2spk):
        res = dict()
        with open(utt2spk, 'r') as f:
            for l in f.readlines():
                res[l.split()[0]] = l.split()[1]
        return res

    def get_spk2xvector(self, spk_xvector_scp: str) -> dict:
        res = kaldiio.load_scp(spk_xvector_scp)
        print(f"Succeed reading xvector from {spk_xvector_scp}")
        return res

    def get_xvector(self, utt):
        xv = self.spk2xvector[self.utt2spk[utt]]
        xv = torch.FloatTensor(xv).squeeze()
        xv = xv / 10  # NOTE: divide by 10 to ensure numerical stability
        return xv

    def get_mel_text_pair(self, utt):
        phn_ids = self.get_text(utt)
        mel = self.get_mel_from_kaldi(utt)
        dur = self.get_dur_from_kaldi(utt)
        xvector = self.get_xvector(utt)

        assert sum(dur) == mel.shape[1], \
            f"Frame length mismatch: utt {utt}, dur: {sum(dur)}, mel: {mel.shape[1]}"
        assert len(dur) == len(phn_ids), \
            f"Duration length and text length mismatch :utt {utt}, dur: {len(dur)}, text: {len(phn_ids)}"

        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "dur": dur,
            "xvector": xvector,
        }
        return res


class XvectorLoaderWithPE(XvectorLoader):
    def __init__(self, utts: str, hparams, feats_scp: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk_name: str, spk_xvector_scp: str, var_scp:str):
        """
        This loader loads speaker ID together with variance (4-dim pitch, 1-dim energy)
        """
        super(XvectorLoaderWithPE, self).__init__(utts, hparams, feats_scp, utt2phns, phn2id,
                 utt2phn_duration, utt2spk_name, spk_xvector_scp)
        self.utt2var = self.get_utt2var(var_scp)

    def get_utt2var(self, utt2var: str) -> dict:
        res = kaldiio.load_scp(utt2var)
        print(f"Succeed reading feats from {utt2var}")
        return res

    def get_var_from_kaldi(self, utt):
        # returns raw log pitch and energy
        # var comprises pov, pitch feature, delta pitch feature, and raw log pitch, and energy. See notion.
        var = self.utt2var[utt]
        var = torch.FloatTensor(var).squeeze()
        assert 5 in var.shape
        if var.shape[0] == 5:
            return var[3, :], var[4, :]
        else:
            return var.T[3, :], var.T[4, :]

    def get_mel_text_pair(self, utt):
        phn_ids = self.get_text(utt)
        mel = self.get_mel_from_kaldi(utt)
        dur = self.get_dur_from_kaldi(utt)
        xvector = self.get_xvector(utt)
        pitch, energy = self.get_var_from_kaldi(utt)

        assert sum(dur) == mel.shape[1] == pitch.shape[0] == energy.shape[0], \
            f"Frame length mismatch: utt {utt}, dur: {sum(dur)}, mel: {mel.shape[1]}, pitch: {pitch.shape[1]}"
        assert len(dur) == len(phn_ids), \
            f"Duration length and text length mismatch :utt {utt}, dur: {len(dur)}, text: {len(phn_ids)}"

        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "dur": dur,
            "xvector": xvector,
            "pitch": pitch,
            "energy": energy
        }
        return res

