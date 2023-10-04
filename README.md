# laplace-lora
Bayesian low-rank adaptation for large language models

This library is largely based on [ASDL](https://github.com/kazukiosawa/asdl/tree/master) and [Laplace](https://github.com/aleximmer/Laplace).

Before installing laplace-lora, first install ASDL from source
```
pip install git+https://github.com/kazukiosawa/asdl
```

To install `laplace-lora`, change directory to `laplace-lora` and run 
```
pip install -e.
```

To fine-tune LlaMA2 or any GPT-like model on common sense reasoning tasks, use 
```
accelerate launch run_gpt.py
``` 
or the bash file 
```
bash run_gpt.bash
``` 
for submission to a slurm server. Customize training arguments like `lora_alpha`, `lora_r`, `lora_dropout`, etc. Set `testing_set` argument to `val` if using the full training set; set `testing_set` argument to `train_val` to split the training set into training and validation set.

To run post-hoc Laplace approximation on saved checkpoints, use 
``` 
accelerate launch run_gpt_laplace.py
``` 
or the bash file 
```
bash run_gpt_laplace.bash
``` 
for submission to a slurm server. Set `laplace_sub` to `all` to use full Laplace-LoRA; set `laplace_sub` to `last_layer` to use last layer Laplace-LoRA. Similarly, set  `testing_set` argument to `val` to use Laplace model evidence on the full training set for optimizing Laplace prior precision; set `testing_set` argument to `train_val` to use minibatch gradient descent on the validation negative log-likelihood for optimizing Laplace prior precision.