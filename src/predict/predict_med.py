import os
import argparse
import asyncio
import aiohttp
import json
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import soundfile as sf
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

    return recordings_df[~recordings_df["current_path"].isin(dst_files)]

    #return list(set(src_files) - set(dst_files))

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

def _build_timestmap_df(mean_predictions, G_X, U_X, time_to_sample, det_threshold):
    """Use the predictions to build an array of contiguous timestamps where the
    probability of detection is above threshold"""
    
    # find where the average 2nd element (positive score) is > threshold
    condition = mean_predictions[:, 1] > det_threshold
    preds_list = []
    current_offset = 0
    for start, stop in _contiguous_regions(condition):
        # start and stop are frame indexes
        # so multiply by n_hop and step_size samples
        # then div by sample rate to get seconds
        start_time = round(start * time_to_sample,2)
        end_time = round(stop * time_to_sample,2)
        preds_list.append({"start": str(start_time), "stop": str(end_time),
                           "med_prob": "{:.4f}".format(
                               np.mean(mean_predictions[start:stop][:, 1]))
                           , "PE":
                           "{:.4f}".format(np.mean(G_X[start:stop])),
                           "MI": "{:.4f}".format(np.mean(U_X[start:stop])),
                           "msc_start": current_offset,
                           "msc_stop": current_offset + (end_time - start_time)})
        current_offset += (end_time - start_time)

    return pd.DataFrame(preds_list)

def _contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx

def plot_mids_MI(X_CNN, y, MI, p_threshold, output_filename, out_format='.png'):
    '''Produce plot of all mosquito detected above a p_threshold. Supply Mutual Information values MI, feature inputs 
    X_CNN, and predictions y (1D array of 0/1s). Plot to be displayed on dashboard either via svg or as part of a
    video (looped png) with audio generated for this visual presentation.
    
    `out_format`: .png, or .svg
    
    '''
    pos_pred_idx = np.where(y>p_threshold)[0]

    fig, axs = plt.subplots(2, sharex=True, figsize=(10,5), gridspec_kw={
           'width_ratios': [1],
           'height_ratios': [2,1]})
    # x_lims = mdates.date2num(T)
    # date_format = mdates.DateFormatter('%M:%S')
    # axs[0].xaxis_date()
    # axs[0].xaxis.set_major_formatter(date_format)
    
    axs[0].set_ylabel('Frequency (kHz)')
    
    axs[0].imshow(np.hstack(X_CNN.squeeze()[pos_pred_idx]), aspect='auto', origin='lower',
                  extent = [0, len(pos_pred_idx), 0, 4], interpolation=None)
    axs[1].plot(y[pos_pred_idx], label='Probability of mosquito')
    axs[1].plot(MI[pos_pred_idx], '--', label='Uncertainty of prediction')
    axs[1].set_ylim([0., 1.02])
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              frameon=False, ncol=2)
    # axs[1].xaxis.set_major_formatter(date_format)
    
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    # axs[1].set_xlim([t[0], t[-1]])
    axs[1].grid(which='major')
    # axs[1].set_xlabel('Time (mm:ss)')
    axs[1].xaxis.get_ticklocs(minor=True)
    axs[1].yaxis.get_ticklocs(minor=True)
    axs[1].minorticks_on()
    labels = axs[1].get_xticklabels()
    # remove the first and the last labels
    labels[0] = ""
    # set these new labels
    axs[1].set_xticklabels(labels)
#     

    plt.subplots_adjust(top=0.985,
    bottom=0.1,
    left=0.0,
    right=0.945,
    hspace=0.065,
    wspace=0.2)
#     plt.show()
    plt.savefig(output_filename, transparent=False)
    plt.close(plt.gcf()) # May be better to re-write to not use plt API
# fig.autofmt_xdate()


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
    det_threshold = 0.5

    # get list of unprocessed wav files from MED process
    recording_list = get_recordings_to_process(args.csv, args.dst)
    logging.debug(f"wav list len: {len(recording_list)}")
    # loop over wav file list
    for _, rec_row in tqdm(recording_list.iterrows()):
        rec_file = rec_row["current_path"]
        logging.debug(f"Recordings File: {rec_file}")
        # get an nd.array for each wav_file
        signal = get_audio_segments(rec_file, rate)
        # run predict on wav file to get dict of offsets and predictions
        predictions = asyncio.run(predict_sample(signal, min_length, rate, win_size, step_size, n_hop))
        # convert JSON output to 2-D array
        # {0: {'1': 0.7850480079650879, '0': 0.21495196223258972},
        #  1: {'1': 0.69720059633255, '0': 0.30279943346977234},
        #  2: {'1': 0.7008734345436096, '0': 0.29912662506103516},
        #  3: {'1': 0.6336677670478821, '0': 0.3663322329521179},
        predictions_array = np.array([[pred['0'], pred['1']] for ind, pred in predictions.items()])
        # layer sliding windows together
        predictions_array_samples = np.array([predictions_array[:-4], predictions_array[1:-3], predictions_array[2:-2]])

        frame_count = signal.unfold(1, win_size * n_hop, step_size * n_hop).shape[1]
        G_X, U_X, _ = active_BALD(np.log(predictions_array_samples), frame_count, 2)
        mean_predictions = np.mean(predictions_array_samples, axis=0)
        
        timestamp_df = _build_timestmap_df(mean_predictions, G_X, U_X, (n_hop * step_size / rate), det_threshold)
        # new output dir
        new_output_dir = Path(args.dst)
        new_output_dir.mkdir(parents=True, exist_ok=True)
        # CSV filename
        csv_output_filename = Path(rec_file).with_suffix(".csv").name
        # save CSV file out
        output_df = timestamp_df.copy()
        output_df["datetime_recorded"] = rec_row["datetime_recorded"]
        output_df["uuid"] = rec_row["uuid"]
        output_df["original_recording"] = rec_row["current_path"]
        output_df.to_csv(Path(new_output_dir, csv_output_filename), index=False)
        # audacicy filename
        txt_output_filename = Path(rec_file).with_name(Path(rec_file).stem+"_mozz_pred.txt").name
        # save audacity file
        audacity_output = [[row["start"], row["stop"], f"{row['med_prob']}, PE: {row['PE']} MI: {row['MI']}"] for ind, row in timestamp_df.iterrows()]
        np.savetxt(Path(new_output_dir, txt_output_filename), audacity_output, fmt='%s', delimiter='\t')
        # audio file name
        audio_output_filename = Path(rec_file).with_name(Path(rec_file).stem+"_mozz_pred.wav").name
        # save audio file
        mozz_audio_list = [signal[0][int(float(row["start"]) * rate):int(float(row["stop"]) * rate)] for ind, row in timestamp_df.iterrows()]
        sf.write(Path(new_output_dir, audio_output_filename), np.hstack(mozz_audio_list), rate)
        # # plot filename
        plot_filename = Path(rec_file).with_suffix(".png").name
        # save png
        # plot_mids_MI(spectrograms, mean_predictions[:,1], U_X, det_threshold, Path(new_output_dir, plot_filename))


        # audio_output_filename, audio_length, has_mosquito = _write_audio_for_plot(text_output_filename, signal, output_filename, root_out, sr)
        #     if has_mosquito:
        #         plot_filename = plot_mids_MI(spectrograms, mean_predictions[:,1], U_X, det_threshold, root_out, output_filename)
        #         _write_video_for_dash(plot_filename, audio_output_filename, audio_length, root_out, output_filename)
