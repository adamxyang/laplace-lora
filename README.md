# laplace-lora
Bayesian low-rank adaptation for large language models

To install `laplace-lora`, run `python setup.py`

To fine-tune LlaMA2 or any GPT-like model on common sense reasoning tasks, use `run_gpt.py` file or the bash file `run_gpt.bash` for submission to a slurm server. Customize training arguments like `lora_alpha`, `lora_r`, `lora_dropout`, etc. 

To run post-hoc Laplace approximation on checkpoints, use `run_gpt_laplace.py` file or the bash file `run_gpt_laplace.bash` for submission to a slurm server.