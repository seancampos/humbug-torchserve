import os
import argparse
import asyncio
import aiohttp
import json
from pathlib import Path
import librosa
import numpy as np
import shutil
from tqdm.auto import tqdm
from typing import List, Dict

import logging

logging.basicConfig(level=logging.INFO)

def get_wav_list(dir: str) -> List[str]:
    file_list = []

    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name.endswith(".wav"):
                file_list.append(os.path.join(path, name))
    
    return file_list

def get_wavs_to_process(src: str, dst: str) -> List[str]:
    src_files = get_wav_list(src)
    dst_files = get_wav_list(dst)

    return list(set(src_files) - set(dst_files))

def get_audio_segments(file: str, sr: int) -> np.ndarray:
    effects = [["remix", "1"],['gain', '-n'],["highpass", "200"]]
    signal, rate = librosa.load(file, sr=sr)
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


async def predict_sample(signal: np.ndarray, min_duration: float, rate: int) -> dict:
    batch = {}
    torchserve_url = "http://localhost:8080/predictions/midsmscmodel"
    sample_length = int(min_duration * rate)
    async with aiohttp.ClientSession() as session:
        # loop over segments of sample_length
        for start in range(0, signal.shape[0], sample_length):
            signal_window = signal[start:start + sample_length]
            # check is sample is of minimum lengeth
            # if it's too short but > 20% of min length then use mean paddding
            if signal_window.shape[0] < sample_length and signal_window.shape[0] > sample_length * 0.2:
                signal_window = pad_mean(signal_window, sample_length)
            # check that sample is correct length
            if signal_window.shape[0] == sample_length:
                # send to torch serve
                async with session.post(torchserve_url, data=signal_window.tobytes()) as resp:
                    # get result and parse
                    result_text = await resp.text()
                    batch[start] = json.loads(result_text)
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
            row.append(f"{species_key}_P{round(batch[offset][species_key])}")
            rows.append(row)
    
    return rows



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()

    min_length = 1.92
    rate = 8000

    # get list of unprocessed wav files from MED process
    wav_list = get_wavs_to_process(args.src, args.dst)
    logging.debug(f"wav list len: {len(wav_list)}")
    # loop over wav file list
    for wav_file in tqdm(wav_list):
        logging.debug(f"Wav File: {wav_file}")
        # get an nd.array for each wav_file
        signal = get_audio_segments(wav_file, rate)
        # run predict on wav file to get dict of offsets and predictions
        batch = asyncio.run(predict_sample(signal, min_length, rate))
        # convert batch to audacity format
        audacity_ndarray = batch_to_audacity(batch, min_length, rate)
        
        # new output dir
        new_output_dir = Path(os.path.dirname(wav_file).replace(args.src, args.dst))
        new_output_dir.mkdir(parents=True, exist_ok=True)
        # txt filename
        text_output_filename = Path(wav_file).with_suffix(".txt").name
        # save file out
        np.savetxt(Path(new_output_dir, text_output_filename), audacity_ndarray, fmt='%s', delimiter='\t')
        # copy wav to new folder
        shutil.copyfile(wav_file, wav_file.replace(args.src, args.dst))
        #print(json.dumps(batch))
