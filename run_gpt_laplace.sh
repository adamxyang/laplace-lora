g=1
c=2
t=12

model=meta-llama/Llama-2-7b-chat-hf


for task in winogrande_s ARC-Challenge ARC-Easy winogrande_m openbookqa boolq 
do
for seed in 21 42 87
do
for laplace_sub in all last_layer  
do
for laplace_hessian in kron 
do
for laplace_prior in homo 
do
for laplace_optim_step in 100
do
    /user/work/ad20999/infrastructure/blue_pebble/bin/lbatch -m 16 -c $c -g $g -t $t --cmd accelerate launch run_gpt_laplace.py \
    --model_name_or_path $model \
    --task_name $task \
    --seed $seed \
    --laplace_sub $laplace_sub \
    --laplace_hessian $laplace_hessian \
    --laplace_prior $laplace_prior \
    --laplace_optim_step $laplace_optim_step \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 300 \
    --testing_set val \
    --lm_head 
done
done
done
done
done
done