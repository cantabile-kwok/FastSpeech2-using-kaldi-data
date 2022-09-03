import torch
import tools
import yaml
import os
from torch.utils.data import DataLoader
from utils.tools import to_device
from tqdm import tqdm
from kaldiio import WriteHelper

if __name__ == '__main__':
    hps, args = tools.get_hparams_decode()
    logger = tools.get_logger(hps.model_dir, "inference.log")

    train_dataset, collate, model = tools.get_correct_class(hps)
    model = model(hps.data, hps.model)
    val_dataset, _, _ = tools.get_correct_class(hps, train=False)

    which_dataset = val_dataset if args.dataset == "val" else train_dataset
    loader = DataLoader(
        which_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        drop_last=False
    )

    ckpt = tools.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    model, _, _, _ = tools.load_checkpoint(ckpt, model, None)
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    met = False
    feats_dir = os.path.join("synthetic_wavs", os.path.basename(hps.model_dir), "tts_gt_spk" if not args.use_control_spk else "tts_other_spk")
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)
    with torch.no_grad():
        with WriteHelper(f"ark,scp:{os.getcwd()}/{feats_dir}/feats.ark,{feats_dir}/feats.scp") as feats:
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                utts = batch['utt']
                batch = to_device(batch, device)
                # =============Control block ===============
                if met:
                    break
                if args.specify_utt_name is not None:
                    if not utts[0] == args.specify_utt_name:
                        continue
                    else:
                        met = True
                elif idx >= args.max_utt_num:
                    break
                # ==========================================

                x, x_lengths = batch['text_padded'].to(device), batch['input_lengths'].to(device)
                if not hps.xvector:
                    if args.use_control_spk:
                        spk = torch.LongTensor([args.control_spk_id]).to(device)
                    else:
                        spk = batch['spk_ids'].to(device)
                else:
                    if args.use_control_spk:
                        spk = which_dataset.spk2xvector[args.control_spk_name]
                        spk = torch.FloatTensor(spk).squeeze().unsqueeze(0).to(device)
                    else:
                        spk = batch['xvector'].to(device)

                output = model(
                    speakers=spk,
                    texts=x,
                    src_lens=x_lengths,
                    max_src_len=max(x_lengths).item(),
                )
                mel_pred = output[0].cpu().numpy().squeeze(0)

                if args.use_control_spk:
                    save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]{utts[0]}"
                else:
                    save_utt_name = f"{utts[0]}_with_GT_spk"

                feats(save_utt_name, mel_pred)


