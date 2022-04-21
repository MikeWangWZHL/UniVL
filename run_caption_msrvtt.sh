export CUDA_VISIBLE_DEVICES=0,3
# export N_GPU=2
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "number of GPU: $N_GPU"

# # msrvtt train
# DATATYPE="msrvtt"
# TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
# VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
# DATA_PATH="data/msrvtt/MSRVTT_data.json"
# FEATURES_PATH="data/msrvtt/msrvtt_videos_features.pickle"
# INIT_MODEL="weight/univl.pretrained.bin"
# OUTPUT_ROOT="ckpts"

# python -m torch.distributed.launch --nproc_per_node=2 \
# main_task_caption.py \
# --do_train --num_thread_reader=4 \
# --epochs=5 --batch_size=128 \
# --n_display=20 \
# --train_csv ${TRAIN_CSV} \
# --val_csv ${VAL_CSV} \
# --data_path ${DATA_PATH} \
# --features_path ${FEATURES_PATH} \
# --output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_caption --bert_model bert-base-uncased \
# --do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
# --batch_size_val 32 --visual_num_hidden_layers 6 \
# --decoder_num_hidden_layers 3 --datatype ${DATATYPE} --stage_two \
# --init_model ${INIT_MODEL}

# eval
DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data.json"
FEATURES_PATH="data/msrvtt/msrvtt_videos_features.pickle"
# INIT_MODEL="ckpts/ckpt_msrvtt_caption/pytorch_model.bin.4"
INIT_MODEL="weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=2 \
main_task_caption.py \
--do_eval --num_thread_reader=4 \
--epochs=1 --batch_size=128 \
--n_display=20 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/eval_ckpt_msrvtt_caption_test_1k_pretrained --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 32 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype ${DATATYPE} --stage_two \
--init_model ${INIT_MODEL} \
--using_1k_test_split