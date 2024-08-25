# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
from multiprocessing import Pool, Manager
import numpy as np
from espnet_onnx import Speech2Text
from tqdm import tqdm
import os
import stat
import re
import time
import librosa
from math import ceil
import psutil
from preprocess import Preprocessor
import subprocess
import onnxruntime as ort
import mmap
import ctypes



PRE_PROCESSORS = {}

def find_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname


def dump_result(save_path, res_list,
                flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR):
    if os.path.exists(save_path):
        print("The result file exists! Please remove it and run again.")
        return
    with os.fdopen(os.open(save_path, flags, mode), 'w') as f:
        for res in res_list:
            f.write(res)


def flat(datas):
    res = []
    for i in datas:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def shuffle_list(data_list):
    data_num = len(data_list[0])
    idxes = np.arange(data_num)
    np.random.shuffle(idxes)

    out_list = []
    for data in data_list:
        out_list.append(
            [data[idx] for idx in idxes]
        )
    return out_list


def load_data(data_dir, batch=1):
    data_list = []
    map_names = []
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                data, sr = librosa.load(fullname, sr=16000)
                data_list.append(data)
                map_names.append(os.path.splitext(os.path.basename(fullname))[0])
    return data_list, map_names


def align_data(data_list, target_batch, sample_data):
    batch_num = len(data_list) // target_batch
    left_num = len(data_list) - batch_num * target_batch
    if left_num:
        data_list += sample_data * (target_batch - left_num)


def generate_batch_data(data_list, map_names, batch_num, device_ids, num_process,
                        enable_multiprocess=False, **kwargs):
    def pack_data(datas, names):
        if batch_num == 1:
            return [[data] for data in datas], [[name] for name in names]
        packed_data = []
        packed_name = []
        data_num = len(datas)
        for idx in range(0, data_num, batch_num):
            _d = datas[idx: idx+batch_num]
            _n = names[idx: idx+batch_num]
            packed_data.append(_d)
            packed_name.append(_n)
        return packed_data, packed_name

    data_list = flat(data_list)
    map_names = flat(map_names)
    if len(data_list) != len(map_names):
        raise ValueError("Num of map_names should be same as data_list.")

    sort = kwargs.get("sort")
    if sort:
        sort_impl = kwargs.get("sort_impl")
        if sort_impl is None:
            raise NotImplementedError("Please provide sort imply method.")
        data_list, map_names = sort_impl([data_list, map_names])

    # pack data by batch
    packed_datas, packed_names = pack_data(data_list, map_names)
    preprocessor_name = kwargs.get("preprocessor")
    if preprocessor_name is not None and PRE_PROCESSORS.get(preprocessor_name) is not None:
        processor = PRE_PROCESSORS.get(preprocessor_name)
        if not enable_multiprocess:
            packed_datas = [processor(data) for data in packed_datas]
        else:
            num_cpu = psutil.cpu_count()
            _preprossed_datas = []
            with Pool(num_cpu) as p:
                for _out_data in list(tqdm(p.imap(processor, packed_datas))):
                    _preprossed_datas.append(_out_data)
            packed_datas = _preprossed_datas

    if kwargs.get("shuffle"):
        packed_datas, packed_names = shuffle_list([packed_datas, packed_names])

    # split by process
    data_num = len(packed_datas)
    split_num = ceil(data_num / num_process)
    splited_packs = []
    num_per_device = ceil(data_num / len(device_ids))
    for idx in range(0, data_num, split_num):
        device_id = device_ids[idx // num_per_device]
        end_idx = idx + split_num
        if args.bink_cpu:
            splited_packs.append(
                (idx // split_num, device_id, packed_datas[idx:end_idx], packed_names[idx:end_idx]))
        else:
            splited_packs.append(
                (device_id, packed_datas[idx:end_idx], packed_names[idx:end_idx]))
    return splited_packs


def infer_multi(speech2text, data_pack, mode='default'):
    if args.bink_cpu:
        cpu_id, device_id, data_list, map_names = data_pack
        p = psutil.Process(os.getpid())
        p.cpu_affinity([cpu_id])
    else:
        device_id, data_list, map_names = data_pack
    only_use_decoder = False
    only_use_encoder = False
    if mode == 'decoder':
        only_use_decoder = True
        num_process = args.num_process_decoder
    elif mode == 'encoder':
        only_use_encoder = True
        num_process = args.num_process_encoder
    else:
        num_process = args.num_process
   # proc = "cat /proc/" + str(os.getpid()) + "/maps"
   # os.system(proc)

    sync_num.append(1)
    while (len(sync_num) != num_process):
        # keep sync between multi processes
        time.sleep(0.05)

    sample_num = 0
    res = []
    names_list = []
    cost_time = 0
    for data_idx, datas in tqdm(enumerate(data_list)):
        names = map_names[data_idx]
        feature = speech2text(datas)
        if mode == 'encoder':
            feature = feature[0]
        else:
            feature = [f[0] for f in feature]
        if isinstance(names, list):
            sample_num += len(names)
        else:
            sample_num += 1
        names_list.append(names)
        res.append(feature)
    sync_num.pop()
    if mode == 'encoder':
        cost_time = speech2text.encoder.encoder.time
    else:
        cost_time += speech2text.scorers['decoder'].decoder.time
        cost_time += speech2text.scorers['ctc'].ctc.time
        cost_time += speech2text.scorers['lm'].lm_session.time
    return sample_num, cost_time, res, names_list


def encoder_preprocess(datas):
    return preprocessor(datas)


def decoder_prerocess(feature_list):
    max_len = max([_.shape[1] for _ in feature_list])
    out_list = [
        np.pad(_, ((0, 0), (0, max_len-_.shape[1]), (0,0))) \
        for _ in feature_list
    ]
    return np.concatenate(out_list, axis=0)


def sort_by_feature(datas, sort_axis=1):
    # sort data in flattened data list
    features = datas[0]
    _res = []
    sorted_idxes = sorted(np.arange(len(features)), key=lambda x:features[x].shape[sort_axis])
    for data in datas:
        _res.append(
            [data[i] for i in sorted_idxes]
        )
    return _res


def speech_preprocess(batch, num_process, device_ids, shuffle=False):
    #malloc_va()
    data_list, map_names = load_data(args.dataset_path)
    #import pdb
    #pdb.set_trace()
    data_list, map_names = sort_by_feature([data_list, map_names], sort_axis=0)
    # align data num for batch num
    align_data(data_list, batch, [data_list[0]])
    align_data(map_names, batch, ["Unvalid data"])

    global PRE_PROCESSORS
    PRE_PROCESSORS['encoder'] = encoder_preprocess
    splited_packs = generate_batch_data(data_list, map_names, batch,
                                        device_ids, num_process, shuffle=shuffle,
                                        sort=True, sort_impl=partial(sort_by_feature, sort_axis=0),
                                        enable_multiprocess=True, preprocessor='encoder')
    return splited_packs


def process_unsplited():
    infer_func = partial(infer_multi)
    sample_num = 0
    data_list, map_names = load_data(args.dataset_path)

    splited_packs = speech_preprocess(args.batch, args.num_process, args.device_ids, shuffle=args.shuffle)
    args.num_process = min(len(splited_packs), args.num_process)
    name_list = []
    res = []
    duration = 0
    with Pool(args.num_process) as p:
        for _num, cost, _re, _name in list(p.imap(infer_func, splited_packs)):
            sample_num += _num
            name_list += _name
            duration += cost
            res.extend(_re)

    res = flat(res)
    name_list = flat(name_list)
    res = [res[idx] for idx in range(len(res)) if name_list[idx] != "Unvalid data"]
    name_list = [name for name in name_list if name != "Unvalid data"]
    out_list = [f"{name_list[i]} {res[i]}\n" for i in range(len(res))]
    total_fps = sample_num / duration
    dump_result(args.result_path, out_list)
    print(f"sample_num:{sample_num}")
    print(f"total: {total_fps}wav/second")


def process_splited(device_id : int):
   # malloc_va()
    command = "cat /proc/" + str(os.getpid()) + "/maps"
    os.system(command)
    speech2text_encoder = Speech2Text(
        model_dir=args.model_path,
        providers=["NPUExecutionProvider"],
        device_id=device_id,
        disable_preprocess=True,
        only_use_encoder=True,
        only_use_decoder=False,
        enable_multibatch=True,
        rank_mode=args.rank_encoder_mode
     )

    speech2text_decoder = Speech2Text(
        model_dir=args.model_path,
        providers=["NPUExecutionProvider"],
        device_id=device_id,
        disable_preprocess=True,
        only_use_encoder=False,
        only_use_decoder=True,
        enable_multibatch=True,
        rank_mode=args.rank_encoder_mode
     )

    # encode process
    infer_encoder = partial(infer_multi, mode='encoder')
    sample_num = 0
    features = []

    splited_packs = speech_preprocess(args.batch_encoder, args.num_process_encoder,
                                      args.device_ids, shuffle=args.shuffle)
    args.num_process_encoder = min(len(splited_packs), args.num_process_encoder)
    map_names = []
    cost_en = 0
    #iresult = lib.unmap(start_addr, size)
    #os.system(command)
   # reserverd_range.close()
   # print(type(splited_packs))
   # print(type(splited_packs[0]))
    #with Pool(args.num_process_encoder) as p:
       # for _num, _cost, _fe, _name in list(p.imap(infer_encoder, splited_packs)):
   # print(type(splited_packs))
    _num, _cost, _fe, _name = infer_encoder(speech2text_encoder,*splited_packs)
    sample_num += _num
    map_names += _name
    cost_en += _cost
    features.append(_fe)
    map_names = flat(map_names)
    features = flat(features)
    features = [np.split(f, f.shape[0], 0) for f in features]
    features = flat(features)

    # decoder process
    infer_decoder = partial(infer_multi, mode='decoder')
    PRE_PROCESSORS['decoder'] = decoder_prerocess
    splited_packs = generate_batch_data(features, map_names, args.batch_decoder,
                                        args.device_ids, args.num_process_decoder, shuffle=args.shuffle,
                                        sort=True, sort_impl=sort_by_feature, preprocessor='decoder')
    res = []
    name_list = []
    cost_de = 0
    #args.num_process_decoder = min(len(splited_packs), args.num_process_decoder)
    #with Pool(args.num_process_decoder) as p:
        #for _, _cost, _re, _name in list(p.imap(infer_decoder, splited_packs)):
    _,_cost, _re, _name = infer_decoder(speech2text_decoder,*splited_packs)
    #output = infer_decoder(speech2text_decoder, *splited_packs)
    #print(type(output))
    #print(len(output))
    #print(output)
    #exit(0)
    cost_de += _cost
    name_list += _name
    res.extend(_re)

    res = flat(res)
    name_list = flat(name_list)
    res = [res[idx] for idx in range(len(res)) if name_list[idx] != "Unvalid data"]
    name_list = [name for name in name_list if name != "Unvalid data"]
    out_list = [f"{name_list[i]} {res[i]}\n" for i in range(len(res))]

    file_name = f'om_{device_id}_1.txt'
    dump_result(file_name, out_list)
    script_path = "compute-wer.py"
    char_value = '--char=1'
    verbosity_value = '--v=1'
    text_value = '/data/data_aishell/transcript/aishell_transcript_v0.8.txt'
    result = subprocess.run(['python', script_path, char_value, verbosity_value, text_value, file_name], capture_output=True, text=True)
    overall_wer = []
    line = result.stdout
    match = re.search(r"Overall -> (\d+\.\d+) %", line)
    if match:
        overall_wer = match.group(1)
    #print(type(overall_wer))
    #print(len(overall_wer))
    encoder_duration = cost_en / args.num_process_encoder
    decoder_duration = cost_de / args.num_process_decoder
    total_fps = sample_num / (encoder_duration + decoder_duration)
    print(f"divece id:{device_id}")
    print(f"sample_num:{sample_num}")
    print(f"encoder: {sample_num/encoder_duration}wav/second")
    print(f"decoder: {sample_num/decoder_duration}wav/second")
    print(f"total: {total_fps}wav/second")
    print(f"acc: {100 - float(overall_wer)}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--dataset_path", default='/workspace/ort-musa-workspace/workspace/espnet/test/S0768/', type=str, help="datapath")
    parser.add_argument('--model_path', default="/root/.cache/espnet_onnx/chenao_conformer", type=str, help='path to the om model and config')
    parser.add_argument('--result_path', default="om_all_beam_size_hahha_11.txt", type=str, help='path to result')
    parser.add_argument('--unsplit', action='store_true', help='enable unsplit mode')
    parser.add_argument('--batch', default=4, type=int, help='batch size of asr process')
    parser.add_argument('--batch_encoder', default=4, type=int, help='batch size  of encode process')
    parser.add_argument('--batch_decoder', default=16, type=int, help='batch size of decode process')
    parser.add_argument('--num_process', default=35, type=int, help='num of process')
    parser.add_argument('--num_process_encoder', default=16, type=int, help='num of encode process')
    parser.add_argument('--num_process_decoder', default=17, type=int, help='num of decode process')
    parser.add_argument('--shuffle', default=True, type=bool, help='data shuffle')
    parser.add_argument('--device_ids', default='0', type=str, help='device for TP')
    parser.add_argument('--d_id', default='0', type=str, help='for MUSA infer')
    parser.add_argument('--rank_encoder_mode', default=True, type=bool, help='enable rank mode for encoder model')
    parser.add_argument('--bink_cpu', action='store_true', help='enable bink cpu in multiprocessing mode')
    parser.add_argument('--seed', default=123, type=int, help='seed for data shuffle')

    args = parser.parse_args()
    np.random.seed(args.seed)
    args.device_ids = [int(_) for _ in args.device_ids.split(",")]
    args.d_id = int(args.d_id)
    manager = Manager()
    sync_num = manager.list()
    config_path = os.path.join(args.model_path, "config.yaml")
    preprocessor = Preprocessor(
        config_path,
        rank_mode=True,
        rank_nums=[259, 387, 515, 643, 771, 899, 1027, 1155, 1283, 1411, 1539, 1667, 1795, 1923]
    )

    if not args.unsplit:
        process_splited(args.d_id)
    else:
        process_unsplited()
