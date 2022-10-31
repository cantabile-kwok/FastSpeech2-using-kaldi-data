import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
import tools
from model.optimizer import ScheduledOptim

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(hps):
    print("Prepare training ...")
    logger_text = tools.get_logger(hps.model_dir)
    logger_text.info(hps)

    train_dataset, collate, model = tools.get_correct_class(hps)
    model = model(hps.data, hps.model)
    val_dataset, _, _ = tools.get_correct_class(hps, train=False)

    batch_size = hps.train.optimizer.batch_size

    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate
    )

    # Prepare model
    # model, optimizer = get_model(args, configs, device, train=True)

    model = nn.DataParallel(model)
    model.to(device)
    num_param = tools.get_param_num(model)
    optimizer = ScheduledOptim(
        model, hps.train, hps.model, current_step=0
    )

    try:
        model, optimizer, iteration, epoch_logged = tools.load_checkpoint(tools.latest_checkpoint_path(hps.model_dir, "G_*.pth"), model, optimizer)
        epoch_start = epoch_logged + 1
        print(f"Loaded checkpoint from {epoch_logged} epoch ({iteration}-th iteration), resuming training.")
    except:
        print(f"Cannot find trained checkpoint, begin to train from scratch")
        epoch_start = 1
        iteration = 0

    Loss = FastSpeech2Loss(hps.data.audio.pitch.feature, hps.data.audio.energy.feature).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Init logger

    train_log_path = os.path.join(hps.model_dir, "train")
    val_log_path = os.path.join(hps.model_dir, "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = iteration
    epoch = epoch_start
    grad_acc_step = hps.train["optimizer"]["grad_acc_step"]
    grad_clip_thresh = hps.train["optimizer"]["grad_clip_thresh"]
    total_step = hps.train["step"]["total_step"]
    log_step = hps.train["step"]["log_step"]
    save_step = hps.train["step"]["save_step"]
    synth_step = hps.train["step"]["synth_step"]
    val_step = hps.train["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = iteration
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        for batch in train_loader:
            batch = to_device(batch, device)
            # print(batch['pitch_padded'].shape, batch['mel_padded'].shape)
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
            # print([mat.shape for mat in output])
            # Cal Loss
            losses = Loss(batch, output)
            total_loss = losses[0]

            # Backward
            total_loss = total_loss / grad_acc_step
            total_loss.backward()
            if step % grad_acc_step == 0:
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr()
                optimizer.zero_grad()

            if step % log_step == 0:
                losses = [l.item() for l in losses]
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    *losses
                )

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                outer_bar.write(message1 + message2)

                log(train_logger, step, losses=losses)

            # if step % synth_step == 0:
                # fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                #     batch,
                #     output,
                #     vocoder,
                #     model_config,
                #     preprocess_config,
                # )
                # log(
                #     train_logger,
                #     fig=fig,
                #     tag="Training/step_{}_{}".format(step, tag),
                # )
                # sampling_rate = preprocess_config["preprocessing"]["audio"][
                #     "sampling_rate"
                # ]
                # log(
                #     train_logger,
                #     audio=wav_reconstruction,
                #     sampling_rate=sampling_rate,
                #     tag="Training/step_{}_{}_reconstructed".format(step, tag),
                # )
                # log(
                #     train_logger,
                #     audio=wav_prediction,
                #     sampling_rate=sampling_rate,
                #     tag="Training/step_{}_{}_synthesized".format(step, tag),
                # )

            if step % val_step == 0:
                model.eval()
                message = evaluate(val_loader, model, step, hps, None, None)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                outer_bar.write(message)

                model.train()

            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                        "iteration": step,
                        "epoch_logged": epoch
                    },
                    os.path.join(
                        hps.model_dir,
                        "G_{}.pth".format(step),
                    ),
                )

            if step == total_step:
                quit()
            step += 1
            outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    hps = tools.get_hparams()

    main(hps)
