import kaldiio
from kaldiio import WriteHelper
import pyworld as pw
import numpy as np
from scipy.io.wavfile import read
import os
import argparse
from sklearn.preprocessing import StandardScaler
import torch
import pickle
from tqdm import tqdm
import torch.multiprocessing as torch_mp
import torch.distributed as dist


# def split(list_a, chunk_size):
#     for i in range(0, len(list_a), chunk_size):
#         yield list_a[i:i + chunk_size]


def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def process(rank, n_procs, args, utt2wav, list_of_utt_list):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_procs, rank=rank)

    utt2fr = args.utt2frame
    utt2frame = dict()
    with open(utt2fr, 'r') as fr:
        for line in fr.readlines():
            utt2frame[line.strip().split()[0]] = int(line.strip().split()[1])

    old_pe = kaldiio.load_scp(args.pe)
    old_pe_keys = set(old_pe.keys())

    del utt2fr
    utt2pitch = dict()
    for utt in tqdm(list_of_utt_list[rank]):
        if utt not in utt2frame.keys():
            continue
        if utt not in old_pe_keys:
            continue
        sr, wav = read(utt2wav[utt])
        # print(wav)
        num_f = utt2frame[utt]
        wav = wav / 32768
        pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=12.5)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)  # 1D array
        # pitch = remove_outlier(pitch)
        utt2pitch[utt] = pitch[:num_f]
    with open(f"{args.target_dir}/pyworld_pitch.{rank}.pkl", 'wb') as fw:
        pickle.dump(utt2pitch, fw)


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, help='wav.scp')
    parser.add_argument("--target-dir", type=str, help='will write to pyworld_pe.ark, pyworld_pe.scp in target dir')
    parser.add_argument('--utt2frame', type=str, help='utt2num_frames file. get mel length')
    parser.add_argument("--pe", type=str, help='pe scp file. Energy index is specified by args.energy_index')
    parser.add_argument("--energy-index", type=int, default=1, help='which index do we use to find energy in pe file')
    parser.add_argument("--nj", type=int, default=10, help='number of parallel jobs')
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    wav_scp = args.wav
    utt2wav = dict()
    with open(wav_scp, 'r') as fr:
        for line in fr.readlines():
            utt2wav[line.strip().split()[0]] = line.strip().split()[1]
    utt_list = list(utt2wav.keys())
    utt_list_split = np.array_split(utt_list, args.nj)
    utt_list_split = [p.tolist() for p in utt_list_split]

    torch_mp.spawn(process, nprocs=args.nj, args=(args.nj, args, utt2wav, utt_list_split))
    print("After the multiprocess...")
    utt2pitch = dict()
    for rank in range(args.nj):
        with open(f"{args.target_dir}/pyworld_pitch.{rank}.pkl", 'rb') as fr:
            utt2pitch_rank = pickle.load(fr)
        utt2pitch.update(utt2pitch_rank)
        os.remove(f"{args.target_dir}/pyworld_pitch.{rank}.pkl")

    pitch_scaler = StandardScaler()
    print("Stat pitch mean and std...")
    for utt, pitch in utt2pitch.items():
        pitch = remove_outlier(pitch)
        if len(pitch) == 0:
            print(f"Utt {utt} pitch length 0 after removing outlier. Skipping it for mean/variance stating.")
            continue
        pitch_scaler.partial_fit(pitch.reshape(-1, 1))
    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
    old_pe = kaldiio.load_scp(args.pe)

    print("Dumping pitch and stat max&min.")
    pitch_max, pitch_min = -np.inf, np.inf
    with WriteHelper(f"ark,scp:{os.getcwd()}/{args.target_dir}/pyworld_pitch_kaldi_energy.ark,"
                     f"{args.target_dir}/pyworld_pitch_kaldi_energy.scp") as feat:
        for utt, p in utt2pitch.items():
            new_p = (p - pitch_mean) / pitch_std
            feat(utt, np.stack([
                        new_p,
                        old_pe[utt][:, args.energy_index]
                    ], axis=1)
                 )
            pitch_max = max(pitch_max, np.max(new_p))
            pitch_min = min(pitch_min, np.min(new_p))

    print(f"Done\n"
          f"Please use {args.target_dir}/pyworld_pitch_kaldi_energy.scp in config\n"
          f"The pitch, energy dim should be set to 0, 1\n"
          f"Also, insert {pitch_min, pitch_max} as the min and max values of pitch.")
