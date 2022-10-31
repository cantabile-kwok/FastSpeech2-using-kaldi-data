import json
import argparse
import os

import kaldiio
import numpy as np
import yaml
from tqdm import  tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_conf = config['data']
    pitch_energy_dims = data_conf['pitch_energy_dims']
    if not data_conf['is_log_pitch']:
        pitch_min, pitch_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf
        max_frame_len = 1000
        for entry in ["train_var_scp"]:
            scp = kaldiio.load_scp(data_conf[entry])
            for key in tqdm(scp.keys(), total=len(list(scp.keys()))):
                var = scp[key]
                shape = var.shape
                frame_len = shape[0]
                max_frame_len = max(frame_len, max_frame_len)
                p, e = var.T[pitch_energy_dims[0], :], var.T[pitch_energy_dims[1], :]
                pitch_min = min(p.min(), pitch_min)
                pitch_max = max(p.max(), pitch_max)
                energy_min = min(e.min(), energy_min)
                energy_max = max(e.max(), energy_max)
        print(pitch_min, pitch_max, energy_min, energy_max)
        stat = [pitch_min, pitch_max, energy_min, energy_max]
        stat = list(map(float, stat))
        data_conf['stat'] = stat
        config['data'] = data_conf
    else:
        pitch_min, pitch_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf
        with kaldiio.WriteHelper(f"ark,scp:"
                                 f"{os.path.dirname(os.path.abspath(data_conf['train_var_scp']))}/processed_pe.ark,"
                                 f"{os.path.dirname(os.path.abspath(data_conf['train_var_scp']))}/processed_pe.scp") as writer:
            max_frame_len = 1000
            cur_raw_pitches = []
            cur_raw_energies = []
            # first pass compute pitch & energy stats
            for entry in ["train_var_scp"]:
                scp = kaldiio.load_scp(data_conf[entry])
                for key in tqdm(scp.keys(), total=len(list(scp.keys()))):
                    var = scp[key]
                    shape = var.shape
                    frame_len = shape[0]
                    max_frame_len = max(frame_len, max_frame_len)
                    p, e = var.T[pitch_energy_dims[0], :], var.T[pitch_energy_dims[1], :]
                    cur_raw_pitches.append(np.exp(p))
                    cur_raw_energies.append(e)
            cur_raw_pitches = np.concatenate(cur_raw_pitches)
            cur_raw_energies = np.concatenate(cur_raw_energies)
            pitch_mean, pitch_std = np.mean(cur_raw_pitches), np.std(cur_raw_pitches)
            energy_mean, energy_std = np.mean(cur_raw_energies), np.std(cur_raw_energies)

            # second pass dump and save
            met_utts = set()
            pitch_min, pitch_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf

            for entry in ['train_var_scp', 'val_var_scp']:
                scp = kaldiio.load_scp(data_conf[entry])
                for key in tqdm(scp.keys(), total=len(list(scp.keys()))):
                    if key in met_utts:
                        continue
                    else:
                        met_utts.add(key)
                    var = scp[key]
                    normalized_pitch = (np.exp(var.T[pitch_energy_dims[0], :]) - pitch_mean) / pitch_std
                    normalized_energy = (var.T[pitch_energy_dims[1], :] - energy_mean) / energy_std
                    pitch_min = min(pitch_min, min(normalized_pitch))
                    pitch_max = max(pitch_max, max(normalized_pitch))
                    energy_min = min(energy_min, min(normalized_energy))
                    energy_max = max(energy_max, max(normalized_energy))
                    writer(key, np.stack([normalized_pitch, normalized_energy], axis=1))

            data_conf['pitch_energy_dims'] = [0, 1]
            original_scp_dirpath = os.path.dirname(os.path.abspath(data_conf['train_var_scp']))
            data_conf['train_var_scp'] = f"{original_scp_dirpath}/processed_pe.scp"
            data_conf['val_var_scp'] = f"{original_scp_dirpath}/processed_pe.scp"
            data_conf['is_log_pitch'] = False

            print(pitch_min, pitch_max, energy_min, energy_max)
            stat = [pitch_min, pitch_max, energy_min, energy_max]
            stat = list(map(float, stat))
            data_conf['stat'] = stat
            config['data'] = data_conf

    # ========== n vocab in model ============
    with open(data_conf['phn2id'], 'r') as f:
        max_value = 0
        for l in f.readlines():
            _, v = l.strip().split()
            v = int(v)
            max_value = max(v, max_value)
    config['model']['n_vocab'] = max_value + 1
    config['model']['max_seq_len'] = max_frame_len + 1

    with open(args.config, 'w') as f:
        yaml.dump(config, f, indent=4)

    print("Finished")

