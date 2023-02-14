import os
import argparse
import asyncio
import aiohttp
import json
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import shutil
import torch
import torchaudio
from tqdm.auto import tqdm
from typing import List, Dict

import logging

logging.basicConfig(level=logging.INFO)

def get_recording_list(dir: str) -> List[str]:
    file_list = []

    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name.endswith(".wav") or name.endswith(".aac"):
                file_list.append(os.path.join(path, name))
    
    return file_list

def get_recordings_to_process(src: str, dst: str) -> List[str]:
    recordings_df = pd.read_csv(src)
    src_files = recordings_df["current_path"].tolist()
    dst_files = get_recording_list(dst)

    return list(set(src_files) - set(dst_files))

def get_audio_segments(file: str, sr: int) -> np.ndarray:
    effects = [["remix", "1"],['gain', '-n'],["highpass", "200"]]
    signal, rate = librosa.load(file, sr=sr)
    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(signal).expand([2, -1]), sample_rate=rate, effects=effects)
    f = waveform[0]
    mu = torch.std_mean(f)[1]
    st = torch.std_mean(f)[0]
    # clip amplitudes
    signal = torch.clamp(f, min=mu-st*3, max=mu+st*3).unsqueeze(0)
    # signal_length += len(signal[0]) / sr
    return signal

# This function pads a short-audio tensor with its mean to ensure that it becomes a 1.92 sec long audio equivalent
def pad_mean(x_temp: np.ndarray, sample_length: int) -> np.ndarray:
    logging.debug("inside padding mean...")
    x_mean = np.mean(x_temp)
    #x_mean.cuda()
    
    logging.debug("X_mean = " + str(x_mean))
    left_pad_amt = int((sample_length - x_temp.shape[0]) // 2)
    logging.debug("left_pad_amt = " + str(left_pad_amt))
    left_pad = np.zeros([left_pad_amt]) #+ (0.1**0.5)*torch.randn(1, left_pad_amt)
    logging.debug("left_pad shape = " + str(left_pad.shape))
    left_pad_mean_add = left_pad + x_mean
    logging.debug("left_pad_mean shape = " + str(left_pad_mean_add))
    logging.debug("sum of left pad mean add = " + str(np.sum(left_pad_mean_add)))
    
    right_pad_amt = int(sample_length - x_temp.shape[0] - left_pad_amt)
    right_pad = np.zeros([right_pad_amt])# + (0.1**0.5)*torch.randn(1, right_pad_amt)
    logging.debug("right_pad shape = " + str(right_pad.shape))
    right_pad_mean_add = right_pad + x_mean
    logging.debug("right_pad_mean shape = " + str(right_pad_mean_add))
    logging.debug("sum of right pad mean add = "  + str(np.sum(right_pad_mean_add)))
    
    f = np.hstack([left_pad_mean_add, x_temp, right_pad_mean_add])
    # f = np.unsqueeze(dim = 0)
    #print("returning a tensor of shape = " + str(f.shape))
    return(f)

def active_BALD(out, X, n_classes):
    if type(X) == int:
        frame_cnt = X
    else:
        frame_cnt = X.shape[0]

    log_prob = np.zeros((out.shape[0], frame_cnt, n_classes))
    score_All = np.zeros((frame_cnt, n_classes))
    All_Entropy = np.zeros((frame_cnt,))
    for d in range(out.shape[0]):
        #  print ('Dropout Iteration', d)
        #  params = unflatten(np.squeeze(out[d]),layer_sizes,nn_weight_index)
        log_prob[d] = out[d]
        soft_score = np.exp(log_prob[d])
        score_All = score_All + soft_score
        # computing F_X
        soft_score_log = np.log2(soft_score + 10e-15)
        Entropy_Compute = - np.multiply(soft_score, soft_score_log)
        Entropy_Per_samp = np.sum(Entropy_Compute, axis=1)
        All_Entropy = All_Entropy + Entropy_Per_samp

    Avg_Pi = np.divide(score_All, out.shape[0])
    Log_Avg_Pi = np.log2(Avg_Pi + 10e-15)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    G_X = Entropy_Average_Pi
    Average_Entropy = np.divide(All_Entropy, out.shape[0])
    F_X = Average_Entropy
    U_X = G_X - F_X
# G_X = predictive entropy
# U_X = MI
    return G_X, U_X, log_prob


async def predict_sample(signal: np.ndarray, min_duration: float, rate: int, win_size: int, step_size: int, n_hop: int) -> dict:
    batch = {}
    torchserve_url = "http://localhost:8080/predictions/midsmedmodel"
    # padding is the difference between the win size and step size
    # The first window is silence prepended to the step size to fill the window
    # then the window slides by the step amount until the last frame is the step followed by
    # silence to fill the window
    pad_amt = (win_size - step_size) * n_hop
    pad_l = torch.zeros(1, pad_amt) + (0.1**0.5) * torch.randn(1, pad_amt)
    pad_r = torch.zeros(1, pad_amt) + (0.1**0.5) * torch.randn(1, pad_amt)
    padded_stepped_signal = torch.cat([pad_l, signal, pad_r], dim=1).unfold(
        1, win_size * n_hop, step_size * n_hop).transpose(0, 1)#.to(device)  # b, 1, s
    async with aiohttp.ClientSession() as session:
        # loop over segments of sample_length
        for batch_index, signal_window in enumerate(padded_stepped_signal):
            # send to torch serve
            async with session.post(torchserve_url, data=signal_window.numpy().tobytes()) as resp:
                # get result and parse
                result_text = await resp.text()
                batch[batch_index] = json.loads(result_text)
    return batch


def batch_to_audacity(batch: Dict, min_duration: float, rate: int) -> List[List[str]]:
    offsets = sorted([int(k) for k in batch.keys()])
    rows = []
    for offset in offsets:
        row = []
        max_species = max(batch[offset], key=batch[offset].get)
        max_prob = batch[offset][max_species]

        row.append(str(offset / rate)) # start
        row.append(str(offset / rate + min_duration)) # end

        row.append(f"Most_likely->{max_species}_P{round(max_prob,3)}")
        rows.append(row)

        species_keys = sorted([s for s in batch[offset].keys()])
        for species_key in species_keys:
            row = []
            row.append(str(offset / rate)) # start
            row.append(str(offset / rate + min_duration)) # end
            row.append(f"{species_key}_P{round(batch[offset][species_key],3)}")
            rows.append(row)
    
    return rows



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument("--csv", help="CSV extract from database",)
    parser.add_argument("--dst", help="Destination directory for output files")
    args = parser.parse_args()

    min_length = 1.92
    rate = 8000
    win_size = 360
    step_size = 120
    n_hop = 128

    # get list of unprocessed wav files from MED process
    recording_list = get_recordings_to_process(args.csv, args.dst)
    logging.debug(f"wav list len: {len(recording_list)}")
    # loop over wav file list
    for rec_file in tqdm(recording_list):
        logging.debug(f"Recordings File: {rec_file}")
        # get an nd.array for each wav_file
        signal = get_audio_segments(rec_file, rate)
        # run predict on wav file to get dict of offsets and predictions
        predictions = asyncio.run(predict_sample(signal, min_length, rate, win_size, step_size, n_hop))

        frame_count = signal.unfold(1, win_size * n_hop, step_size * n_hop).shape[1]
        G_X, U_X, _ = active_BALD(np.log(predictions), frame_count, 2)
        mean_predictions = np.mean(predictions, axis=0)
        
        # new output dir
        new_output_dir = Path(os.path.dirname(rec_file).replace(args.src, args.dst))
        new_output_dir.mkdir(parents=True, exist_ok=True)
        # txt filename
        text_output_filename = Path(rec_file).with_suffix(".txt").name
        # save file out
        pd.DataFrame(mean_predictions).to_csv(Path(new_output_dir, text_output_filename))
        # np.savetxt(Path(new_output_dir, text_output_filename), audacity_ndarray, fmt='%s', delimiter='\t')
        # copy wav to new folder
        # shutil.copyfile(rec_file, rec_file.replace(args.src, args.dst))
        #print(json.dumps(batch))
