CUDA_VISIBLE_DEVICES=0 python hrpo_gsm8k.py --model_name Qwen/Qwen2.5-3B-Instruct --residual_r_min 0.98 --group_size 8

CUDA_VISIBLE_DEVICES=0 python eval_gsm8k.py --checkpoint_path PATH/TO/CHECKPOINT --batch_size BATCH_SIZE --greedy