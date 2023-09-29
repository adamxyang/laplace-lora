import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM, LlamaTokenizer
)

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModel,
    PeftConfig
)

from laplace import Laplace
import pickle
import dill

logger = get_logger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='winogrande_s',
        help="The name of the glue task to train on.",
        # choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./outputs', help="Where to store the final model.")
    parser.add_argument("--peft_method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=21, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='1000',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        default=True,
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--load_step", type=int, default=999)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--laplace_hessian", type=str, default='kron')
    parser.add_argument("--laplace_sub", type=str, default='last_layer')
    parser.add_argument("--laplace_prior", type=str, default='homo', help='homo, hetero')
    parser.add_argument("--laplace_optim_step", type=int, default=1000)
    parser.add_argument("--testing_set", type=str, default='train_val')
    parser.add_argument("--laplace_predict", type=str, default='mc_corr_100000', help='probit bridge bridge_norm mc_indep mc_corr')
    parser.add_argument("--lm_head", action="store_true", default=False)
    args = parser.parse_args()

    print(args)

    peft_method = 'lora'
    if args.lm_head:
        peft_method = 'lora_lmhead'
    if args.testing_set != 'val':
        peft_method += args.testing_set
    
        

    args.output_dir += f'/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}'
    args.laplace_output_dir = f'outputs_laplace/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}/'
    

    if 'last_layer' in args.laplace_sub:
        assert args.laplace_prior == 'homo'

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main(load_step):
    args = parse_args()
    args.load_step = load_step
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    laplace_output_dir = args.laplace_output_dir + f'step_{args.load_step}'
    os.makedirs(laplace_output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
        raw_datasets = load_dataset("glue", args.task_name)
    elif args.task_name in ['cb', 'wic', 'boolq']:
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif 'ARC' in args.task_name:
        raw_datasets = load_dataset('ai2_arc', args.task_name)
    elif 'winogrande' in args.task_name:
        raw_datasets = load_dataset('winogrande', args.task_name)
    else:
        raw_datasets = load_dataset(args.task_name)

    if 'ARC' in args.task_name or 'openbookqa' in args.task_name:
        # Initialize counters
        count_3_choices_train = 0
        count_5_choices_train = 0
        count_3_choices_valid = 0
        count_5_choices_valid = 0

        # Count in the training dataset
        for example in raw_datasets["train"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_train += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_train += 1

        # Count in the validation dataset
        for example in raw_datasets["validation"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_valid += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_valid += 1

        # Get total counts
        total_train = len(raw_datasets["train"])
        total_valid = len(raw_datasets["validation"])

        # Print counts
        print('====counts train====')
        print(f"Total number of training examples: {total_train}")
        print(f"Number of training questions with 3 choices: {count_3_choices_train}")
        print(f"Number of training questions with 5 choices: {count_5_choices_train}")

        print('====counts valid====')
        print(f"Total number of validation examples: {total_valid}")
        print(f"Number of validation questions with 3 choices: {count_3_choices_valid}")
        print(f"Number of validation questions with 5 choices: {count_5_choices_valid}")

        # Filter the examples in the training dataset
        filtered_train = raw_datasets["train"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the validation dataset
        filtered_valid = raw_datasets["validation"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the test dataset
        filtered_test = raw_datasets["test"].filter(lambda example: len(example['choices']['label']) == 4)

        # Replace the original datasets with the filtered datasets
        raw_datasets["train"] = filtered_train
        raw_datasets["validation"] = filtered_valid
        raw_datasets["test"] = filtered_test

        print('====counts train====')
        print(f"Total number of training examples: {len(raw_datasets['train'])}")
        print('====counts valid====')
        print(f"Total number of validation examples: {len(raw_datasets['validation'])}")

        def convert_choices_to_alpha(example):
            # Define a mapping from numerical to alphabetical labels
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

            # Convert the 'label' field in 'choices'
            example['choices']['label'] = [mapping.get(label, label) for label in example['choices']['label']]

            # Convert the 'answerKey' field
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])

            example['choices']['text'] = [text if text.endswith('.') else text + '.' for text in example['choices']['text']]
            example['choices']['text'] = [text[0].upper() + text[1:] if text else text for text in example['choices']['text']]

            return example

        # Apply the conversion to the training, validation, and test datasets
        raw_datasets["train"] = raw_datasets["train"].map(convert_choices_to_alpha)
        raw_datasets["validation"] = raw_datasets["validation"].map(convert_choices_to_alpha)
        raw_datasets["test"] = raw_datasets["test"].map(convert_choices_to_alpha)

        print('====train data====')
        from collections import Counter

        # Initialize counters for training and validation datasets
        counter_train = Counter()
        counter_valid = Counter()

        # Count in the training dataset
        for example in raw_datasets["train"]:
            counter_train.update(example['answerKey'])

        # Count in the validation dataset
        for example in raw_datasets["validation"]:
            counter_valid.update(example['answerKey'])

        # Print the results
        print("Training dataset counts:")
        for choice, count in counter_train.items():
            print(f"Choice {choice}: {count} occurrences")

        print("Validation dataset counts:")
        for choice, count in counter_valid.items():
            print(f"Choice {choice}: {count} occurrences")


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left', use_auth_token='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')
    tokenizer.pad_token = tokenizer.bos_token
    if args.task_name in ['boolq']: #,'winogrande_m', 'winogrande_s']:
        tokenizer.add_eos_token = True
    

    output_dir = args.output_dir + f'/step_{args.load_step}'


    peft_config = PeftConfig.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, output_dir)
    model.print_trainable_parameters()
    print('======')
    print(model)



    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'lora' in name:
            if 'all' in args.laplace_sub:
                param.requires_grad = True

    model.print_trainable_parameters()


    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.task_name == 'boolq':
            texts = [f"Answer the question with only True or False: {question} Context: {passage}" for passage, question in zip(examples['passage'], examples['question'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question_stem'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'ARC' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'winogrande' in  args.task_name:
            texts = [f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B {option2}. Answer:" for question, option1, option2 in zip(examples['sentence'], examples['option1'], examples['option2'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"1": 0, "2": 1, "":None}
            result["labels"] = [map_dict[label] for label in examples["answer"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    processed_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    if args.testing_set == 'test':
        ds = processed_dataset.train_test_split(test_size=0.5, seed=42, shuffle=False)
        val_dataset, eval_dataset = ds["train"], ds["test"]
    elif args.testing_set == 'train_val':
        ds = train_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
        train_dataset, val_dataset = ds["train"], ds["test"]
        eval_dataset = processed_dataset
    else:
        eval_dataset = processed_dataset

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    # selected_indices = list(range(10))
    # train_dataset = train_dataset.select(selected_indices)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.testing_set != 'val':
        val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    
    class CustomLMHead_lora(torch.nn.Module):
        def __init__(self, original_lm_head, id_list):
            super().__init__()
            self.id_list = id_list
            
            # Trim the lm_head linear weights
            original_weight = original_lm_head.weight[id_list, :].clone()
            self.linear = torch.nn.Linear(in_features=original_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
            self.linear.weight.data = original_weight.to(torch.float32)
            self.linear.weight.requires_grad = False

            self.lora_dropout = original_lm_head.lora_dropout['default']

            original_lora_A_weight = original_lm_head.lora_A["default"].weight.clone()
            self.lora_A = torch.nn.Linear(in_features=original_lora_A_weight.shape[1], out_features=original_lora_A_weight.shape[0], bias=False).to(accelerator.device)
            self.lora_A.weight.data = original_lora_A_weight.to(torch.float32)
            # if args.laplace_sub == 'all':
            self.lora_A.weight.requires_grad = True
            
            # Trim the lora_B weights
            original_lora_B_weight = original_lm_head.lora_B["default"].weight[id_list, :].clone()
            self.lora_B = torch.nn.Linear(in_features=original_lora_B_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
            self.lora_B.weight.data = original_lora_B_weight.to(torch.float32)
            self.lora_B.weight.requires_grad = True
            
            self.scaling = args.lora_alpha / args.lora_r

        def forward(self, x):
            linear_out = self.linear(x[:, -1, :].to(torch.float32))
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x[:, -1, :].to(torch.float32))))
            return linear_out + lora_out * self.scaling


    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

            if args.task_name == 'boolq':
                self.id_list = [tokenizer.encode('False')[1], tokenizer.encode('True')[1]]
            elif args.task_name == 'openbookqa':
                self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
            elif 'ARC' in args.task_name:
                self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
            elif 'winogrande' in args.task_name:
                self.id_list = [tokenizer.encode('A')[1], tokenizer.encode('B')[1]]

            if args.lm_head:
                original_lm_head = model.base_model.model.lm_head
                model.base_model.model.lm_head = CustomLMHead_lora(original_lm_head, self.id_list).to(accelerator.device) 
            
            self.model = model

            model.print_trainable_parameters()
            print(self.model)


        def forward(self, **kwargs):
            kwargs.pop('labels', None)
            output_dict = self.model(**kwargs)
            logits = output_dict['logits']
            if args.lm_head:
                selected_logits = logits
            else:
                selected_logits = logits[:, -1, self.id_list]
            return selected_logits.to(torch.float32)
        
    model = WrappedModel(model)

    print('====model====')
    # print(model.model.base_model.model.lm_head.linear.weight)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    # Prepare everything with our `accelerator`.
    if args.testing_set == 'val':
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, eval_dataloader, lr_scheduler
        )
    model.eval()

    # Get the metric function
    if args.task_name is not None:
        if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
            metric = evaluate.load("glue", args.task_name, experiment_id=f"{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")
        elif args.task_name in ['cb', 'wic', 'boolq']:
            metric = evaluate.load("super_glue", args.task_name, experiment_id=f"{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")
        else:
            metric = evaluate.load("accuracy", experiment_id=f"{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")
    else:
        metric = evaluate.load("accuracy", experiment_id=f"{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")


    if args.laplace_prior == 'hetero':
        la = Laplace(model, 'classification',
                        subset_of_weights='all',
                        hessian_structure=args.laplace_hessian)
    elif args.laplace_prior == 'homo':
        la = Laplace(model, 'classification', prior_precision=1.,
                        subset_of_weights='all',
                        hessian_structure=args.laplace_hessian)
    elif args.laplace_prior == 'hybrid':
        if not args.use_bias:
            prior_prec = torch.ones(2).to(accelerator.device)
        else:
            prior_prec = torch.ones(3).to(accelerator.device)
        la = Laplace(model, 'classification', prior_precision=prior_prec,
                        subset_of_weights='all',
                        hessian_structure=args.laplace_hessian)

    print('----fitting Laplace-----')
    la.fit(train_dataloader)

    if args.testing_set != 'val':
        prior_precision = la.optimize_prior_precision(method='CV', val_loader=val_dataloader, link_approx='mc', log_prior_prec_min=-3, log_prior_prec_max=4, grid_size=50, n_steps=args.laplace_optim_step, lr=1e-1)
    else:
        prior_precision = la.optimize_prior_precision(method='marglik', n_steps=args.laplace_optim_step, lr=1e-1)
        print(f'prior precision: {prior_precision}')
    torch.save(prior_precision, f'{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    print('prior precision', prior_precision)



    samples_seen = 0
    output_dicts = []
    f_mu_list = []
    f_var_list = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu_list.append(f_mu)
            f_var_list.append(f_var)
        
        samples = int(args.laplace_predict.split('_')[-1])
        f_mu = f_mu.expand(samples, -1, -1)
        f_var = f_var.expand(samples, -1, -1, -1)

        logits = f_mu + (torch.linalg.cholesky(f_var + torch.eye(f_var.shape[-1]).to(f_var.device)*1e-6).to(f_mu.dtype) @ torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(accelerator.device)).squeeze(-1)
        logits = torch.softmax(logits, dim=-1).mean(0)
        
        predictions = logits.argmax(dim=-1)

        logits = logits.detach()
        for j in range(logits.size(0)):
            probs = logits[j]  # F.softmax(logits[j], -1) do softmax when evaluating for ECE/NLL
            label = batch["labels"]
            output_dict = {
                'index': args.per_device_eval_batch_size * step + j,
                'true': label[j].item(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
            
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    f_mu = torch.cat(f_mu_list, dim=0)
    f_var = torch.cat(f_var_list, dim=0)
    print('f_mu shape', f_mu.shape)
    print('f_var shape', f_var.shape)
    print(f_mu)
    print(f_var)
    torch.save(f_mu, f'{laplace_output_dir}/f_mu_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    torch.save(f_var, f'{laplace_output_dir}/f_var_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')

    output_path = os.path.join(output_dir, f'eval_res_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_predict}_{args.laplace_optim_step}.json')
    print(f'writing outputs to \'{output_path}\'')

    # delete the file if it exists
    if os.path.isfile(output_path):
        os.remove(output_path)

    with open(output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    eval_metric = metric.compute()

    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}

    all_results_path = os.path.join(output_dir, f"all_results_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_predict}_{args.laplace_optim_step}.json")

    # delete the all_results file if it exists
    if os.path.isfile(all_results_path):
        os.remove(all_results_path)

    
    # write to the all_results file
    with open(all_results_path, "w") as f:
        json.dump(all_results, f)

    del model, train_dataloader, la, f_mu, f_var, f_mu_list, f_var_list, metric, eval_metric, output_dicts, eval_dataloader
    torch.cuda.empty_cache()




if __name__ == "__main__":
    step_list = [0,*list(range(999, 10999, 1000))]
    for load_step in step_list:
        main(load_step)