import json
import argparse
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
    pitch_min, pitch_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf
    max_frame_len = 1000
    for entry in ["train_var_scp"]:
        scp = kaldiio.load_scp(data_conf[entry])
        for key in tqdm(scp.keys(), total=len(list(scp.keys()))):
            var = scp[key]
            shape = var.shape
            frame_len = shape[0] if shape[1] == 5 else shape[1]
            max_frame_len = max(frame_len, max_frame_len)
            p, e = (var[3, :], var[4, :]) if shape[0] == 5 else (var.T[3, :], var.T[4, :])
            pitch_min = min(p.min(), pitch_min)
            pitch_max = max(p.max(), pitch_max)
            energy_min = min(e.min(), energy_min)
            energy_max = max(e.max(), energy_max)
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
