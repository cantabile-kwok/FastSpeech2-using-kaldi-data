import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

# from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
# from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(val_loader, model, step, hps, logger=None, vocoder=None):
    # preprocess_config, model_config, train_config = configs

    # Get dataset
    # dataset = Dataset(
    #     "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    # )
    batch_size = hps.train["optimizer"]["batch_size"]

    # Get loss function
    Loss = FastSpeech2Loss(hps.data.audio.pitch.feature, hps.data.audio.energy.feature).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    sample_num = 0
    for batch in val_loader:
        # for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            if not hps.xvector:
                spk = batch['spk_ids']
            else:
                spk = batch['xvector']

            output = model(
                speakers=spk,
                texts=batch['text_padded'],
                src_lens=batch['input_lengths'],
                max_src_len=max(batch['input_lengths']).item(),
                mel_lens=batch['output_lengths'],
                max_mel_len=max(batch['output_lengths']).item(),
                p_targets=batch['pitch_padded'],
                d_targets=batch['dur_padded'],
                e_targets=batch['energy_padded']
            )

            # Cal Loss
            losses = Loss(batch, output)

            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * len(batch['text_padded'])
            sample_num += len(batch["text_padded"])

    loss_means = [loss_sum / sample_num for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)