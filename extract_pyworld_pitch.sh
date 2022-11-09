# extract pyworld pitch, in multiprocessing way, and store in $target_dir/pyworld_pitch_kaldi_energy.scp and ark.
# Need to pass in utt2num_frames file, which can be obtained by feat-to-len kaldi command.
# Need to pass in a variance.scp or other scp that contains extracted energy. The index of energy is specified.
# The extraction contains normalization, i.e. the saved data is of zero mean and unit variance. Min and max values are printed also.

#python extract_pyworld_pitch.py \
#      --wav filelists/LJ/wav.scp \
#      --target-dir filelists/LJ/pyworld/ \
#      --utt2frame filelists/LJ/utt2num_frames \
#      --pe filelists/LJ/processed_pe.scp \
#      --energy-index 1 \
#      --nj 30

python extract_pyworld_pitch.py \
      --wav filelists/Libri_all/wav.scp \
      --target-dir filelists/Libri_all/pyworld/ \
      --utt2frame filelists/Libri_all/utt2num_frames \
      --pe filelists/Libri_all/processed_pe.scp \
      --energy-index 1 \
      --nj 30