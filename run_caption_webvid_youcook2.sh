export CUDA_VISIBLE_DEVICES=0,3
export N_GPU=2
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

# webvid eval
TRAIN_CSV="data/webvid_1percent/webvid_1percent.csv"
VAL_CSV="data/webvid_1percent/webvid_1percent.csv"
DATA_PATH="data/webvid_1percent/webvid_1percent_caption.pickle"
FEATURES_PATH="data/webvid_1percent/webvid_1percent_features.pickle"
# INIT_MODEL="weight/univl.pretrained.bin"
INIT_MODEL="ckpts/ckpt_msrvtt_caption/pytorch_model.bin.4"
OUTPUT_ROOT="ckpts"

python -m torch.distributed.launch --nproc_per_node=$N_GPU \
main_task_caption.py \
--do_eval --num_thread_reader=4 \
--epochs=1 --batch_size=16 \
--n_display=10 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/eval_ckpt_webvid_1percent_caption_finetuned --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 128 --max_frames 96 \
--batch_size_val 128 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --stage_two \
--init_model ${INIT_MODEL}
