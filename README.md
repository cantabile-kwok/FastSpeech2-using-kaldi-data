# FastSpeech2 using Kaldi data
> Some people is habituated to use Kaldi & ESPnet feature & data. Meanwhile, the training & inference procedures in GlowTTS appears to be charming. So this project modifies [ming024's FastSpeech2 implementation](https://github.com/ming024/FastSpeech2) and [glow-tts official repo](https://github.com/jaywalnut310/glow-tts) so that we can train FastSpeech2 in GlowTTS style while using features extracted from Kaldi & ESPnet.

## Data Preparation
All basic TTS acoustic models need two inputs: text and mel spectrogram. For those who need external duration knowledge, ground truth duration sequences are also a must.
In this project, we use `filelists` directory to store these data.
Specifically:
 * `train_utts.txt` and `val_utts.txt` stores utterance-ids that train set and validation set should use.
 * `text` specifies phoneme sequences for each utterance-id, in `utt-id phone1 phone2 phone3...` format.
 * `phones.txt` gives a mapping from phoneme to integer indexes.
 * `feats.scp` gives a Kaldi-style file pointer to features stored in `ark` format.
 * `phn_duration` specifies integer duration sequences for each utterance-id. The length of it must match `text`. The sum of it must match `feats.scp`. We do this sanity check in our code.

Also, we need to provide speaker information. Depending on tasks, we basically provide two ways to do so:
 1. In case of integer speaker-ids: there should be a `utt2spk.json` file that maps utterance-ids to speaker indexes.
 2. In case of speaker-embedding (xvectors, ivectors, etc.), there should be a `utt2spk` file that maps utterance-ids to speaker names (split with spaces). Then speaker embedding data must also be provided, e.g. `spk_xvector.scp`.

Additional information can be provided the same way, e.g. `var.scp` specifies variance data (pitch, energy) in FastSpeech2 training.

We do not check the size of provided data, e.g. whether `feats.scp` has more utts than `utts.txt`. Only ensure that all the utts listed in `utts.txt` have their corresponding entry in other files.

Then, we specify these files in a configuration yaml for each experiment. Those configurations are often stored in `configs` directory.

## Configuration
Our yaml config files have three parts: **model, data, train**, as can be seen in the example.
We change the file paths in `data` part for different experiments. Note that we explicitly specify each file path so that we can switch to other files flexibly.

Also, these file paths are split into train and val data. We often use a single file (like a combined `feats.scp`) for training and validation dataset, but we have to explicitly specify them both.

Other configurations can be easily understood. Note that we use `xvector` in these configurations to specify whether to use speaker-ids (embedding tables) or pre-extracted speaker embeddings.

## Training
First, run
```shell
name=LJ  # config file basename
python get_variance_range.py -c configs/${name}.yaml
```
This does some preprocessing and updates the yaml configuration. It stores the min and max of pitch and energy values and the maximum length of mel-spectrograms. 
Be careful that your comments in the original yaml file will be discarded in this step.

Simply this line will do the training
```shell
name=LJ
python train.py -c configs/${name}.yaml -m $name
```
Then `logs/$name` dir will be generated to log the experiments (including checkpoints).

The checkpoints will be saved as `G_{iter_num}.pth`.

## Inference
To do TTS inference by default, you can use
```shell
name=LJ
python inference.py -c configs/${name}.yaml -m $name
```
Then `synthetic_wavs/${name}/tts_gt_spk/` will be generated and it contains the synthetic features. As we only care about acoustic models here, we don't provide vocoder interfaces. The saved features is in kaldi ark,scp style.

The inference script also supports some options, like 
```shell
name=LJ
python inference.py -c configs/${name}.yaml -m $name \
      --max-utt-num 100 \
      --use-control-spk \
      --control-spk-id 2
```
controls the synthesized speaker to be index 2. Alternatively, if using xvector, you can use `--control-spk-name some_spk_name` to specify the controlled speaker name.

```shell
name=LJ
python inference.py -c configs/${name}.yaml -m $name \
      --specify-utt-name LJ010-0110
```
will only synthesize this utterance. It can be used together with speaker controlling options.

You can also use `-s` or `--seed` to control random seed, and `--dataset train/val` to select a dataset to be synthesized (the default is val).

---
## Update for pitch processing
Please use a variance.scp **before CMVN**. Then set `is_log_pitch` in conf to True, then run `get_variance_range.py`. 
This will automatically process the variance (i.e. find mean and variance, normalize, and record min/max values) and save to filelists/${name}/processed_pe.scp (and ark).
