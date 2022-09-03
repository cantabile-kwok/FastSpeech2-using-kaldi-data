import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, data_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(data_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            data_config["audio"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f))
        #     self.speaker_emb = nn.Embedding(
        #         n_speaker,
        #         model_config["transformer"]["encoder_hidden"],
        #     )
        n_speaker = model_config['n_speaker']
        self.speaker_emb = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])

    def forward(
        self,
        speakers,  # spk id seq
        texts,  # phn id seq
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # print("pitch shape in model forward")
        # print(p_targets.shape)
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        # if self.speaker_emb is not None:
        output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
        # print("output shape after spk embedding added", output.shape)
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len if max_mel_len is not None else None,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        # print("output shape after variance predictor", output.shape)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        # print("output shape after decoder", output.shape)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )


class FastSpeech2Xvector(nn.Module):
    """ FastSpeech2 """

    def __init__(self, data_config, model_config):
        super(FastSpeech2Xvector, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(data_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            data_config["audio"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f))
        #     self.speaker_emb = nn.Embedding(
        #         n_speaker,
        #         model_config["transformer"]["encoder_hidden"],
        #     )
        self.xvector_linear = nn.Sequential(
            nn.Linear(model_config['xvector_dim'], model_config['xvector_dim']//2),
            nn.ReLU(),
            nn.Linear(model_config['xvector_dim']//2, model_config["transformer"]["encoder_hidden"])
        )

    def forward(
        self,
        speakers,  # spk xvector seq
        texts,  # phn id seq
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        # if self.speaker_emb is not None:
        output = output + self.xvector_linear(speakers).unsqueeze(1).expand(
            -1, max_src_len, -1
        )  # speakers: B, xv_dim.

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
