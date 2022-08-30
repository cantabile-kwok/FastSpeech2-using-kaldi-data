import json
import argparse
import kaldiio
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    data_conf = config['data']
    pitch_min, pitch_max, energy_min, energy_max = np.inf, -np.inf, np.inf, -np.inf

    for entry in ["train_var_scp", "val_var_scp"]:
        scp = kaldiio.load_scp(data_conf[entry])
        for key in scp.keys():
            var = scp[key]
            shape = var.shape
            p, e = (var[3, :], var[4, :]) if shape[0] == 5 else (var.T[3, :], var.T[4, :])
            pitch_min = min(p.min(), pitch_min)
            pitch_max = max(p.max(), pitch_max)
            energy_min = min(e.min(), energy_min)
            energy_max = max(e.max(), energy_max)
    print(pitch_min, pitch_max, energy_min, energy_max)
    stat = [pitch_min, pitch_max, energy_min, energy_max]
    data_conf['stat'] = stat
    config['data'] = data_conf

    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)
    print("Finished")
