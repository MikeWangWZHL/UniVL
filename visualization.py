import os
import csv
import json
import pandas as pd
import numpy as np
import pickle


# load csv, video ids
csv_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/UniVL/data/webvid_1percent/webvid_1percent.csv'
csv_file = pd.read_csv(csv_path, dtype=str)
video_id_list = [itm for itm in csv_file['video_id'].values]
print('video id num:', len(video_id_list))

# load UniVL captions
univl_results_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/UniVL/ckpts/ckpt_webvid_1percent_caption_finetuned_msrvtt/hyp.txt'
univl_captions = []
with open(univl_results_path) as f:
    for line in f:
        univl_captions.append(line.strip())
print('univl caption num:', len(univl_captions))

# load original captions
original_results_path = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/UniVL/ckpts/ckpt_webvid_1percent_caption_finetuned_msrvtt/ref.txt'
original_captions = []
with open(original_results_path) as f:
    for line in f:
        original_captions.append(line.strip())
print('original caption num:', len(original_captions))

# load our captions
gpt_response_dict = '/shared/nas/data/m1/wangz3/video_language_pretraining_project/BLIP_Video_Captioning/cluster_results/processed_gpt3_responses/WebVid/webvid_1_percent_v1.1.json'
gpt_response = json.load(open(gpt_response_dict))


# visualize:
output_path = './visualization_caption/univl_msrvtt_finetuned__gpt3_v1.1__visualization.txt'
lines = []
for i in range(len(video_id_list)):
    video_id = video_id_list[i]
    univl_cap = univl_captions[i]
    original_cap = original_captions[i]
    if video_id in gpt_response:
        gpt_cap = gpt_response[video_id][0]['text'].strip()
        line = f'### {video_id} ###\noriginal:{original_cap}\nunivl:{univl_cap}\nours:{gpt_cap}\n\n' 
        lines.append(line)

with open(output_path, 'w') as out:
    for line in lines:
        out.write(line)