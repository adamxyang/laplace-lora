g=1
c=1
t=6

model=meta-llama/Llama-2-7b-chat-hf

for task in winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq  
do
for seed in 21 42 87
do
    /user/work/ad20999/infrastructure/blue_pebble/bin/lbatch -m 16 -c $c -g $g -t $t --cmd accelerate launch run_gpt.py \
    --model_name_or_path $model \
    --task_name $task \
    --seed $seed \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 300 \
    --testing_set val \
    --lm_head
done
done
