import os

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
import copy

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
)


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='winogrande_s',
        help="The name of the glue task to train on.",
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
        default=400,
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
        default='meta-llama/Llama-2-7b-hf',
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
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=.5,
        help="Gradient clipping norm.",
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
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
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
    parser.add_argument(
        "--save_train_results",
        action="store_true",
        default=False,
        help="Whether or not to save evaluation on training set.",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--testing_set", type=str, default='val')
    parser.add_argument("--lm_head", action="store_true", default=False)
    args = parser.parse_args()

    print(args)

    peft_method = 'lora'
    if args.lm_head:
        peft_method = 'lora_lmhead'
    if args.testing_set != 'val':
        peft_method += args.testing_set

    
    args.output_dir += f'/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}'

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


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
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
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
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
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

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
    if args.task_name in ['boolq']:  #,'winogrande_m', 'winogrande_s']:
        tokenizer.add_eos_token = True
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, load_in_8bit=True
    )


    
    target_modules=['v_proj','q_proj']
    if args.lm_head:
        target_modules.append('lm_head')
    peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)

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

    # print('====train data====')
    train_dataset = processed_datasets["train"]
    # print('====validation data====')
    processed_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    if args.testing_set == 'test':
        ds = processed_dataset.train_test_split(test_size=0.5, seed=42, shuffle=False)
        val_dataset, eval_dataset = ds["train"], ds["test"]
    elif args.testing_set == 'train_val':
        ds = train_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
        train_dataset, val_dataset = ds["train"], ds["test"]
        eval_dataset = processed_dataset
    elif args.testing_set == 'val':
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

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    if args.testing_set != 'val':
        val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Prepare everything with our `accelerator`.

    
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

            self.model = model


        def forward(self, **kwargs):
            kwargs.pop('labels', None)
            output_dict = self.model(**kwargs)
            logits = output_dict['logits']
            selected_logits = logits[:, -1, self.id_list]
            output_dict['logits'] = selected_logits
            return output_dict   
    

    model = WrappedModel(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    parameters_to_clip = [p for group in optimizer.param_groups for p in group['params']]

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
            metric = evaluate.load("glue", args.task_name, experiment_id=f"{args.output_dir}")
        elif args.task_name in ['cb', 'wic', 'boolq']:
            metric = evaluate.load("super_glue", args.task_name, experiment_id=f"{args.output_dir}")
        else:
            metric = evaluate.load('accuracy', experiment_id=f"{args.output_dir}")
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    test_loader_list = [eval_dataloader]
    test_loader_names = ['eval']
    if args.testing_set != 'val':
        test_loader_list.append(val_dataloader)
        test_loader_names.append('val')
        
    for epoch in range(starting_epoch, args.num_train_epochs):
        active_dataloader = train_dataloader
        for step, train_batch in enumerate(active_dataloader):

            if isinstance(checkpointing_steps, int):
                for test_loader, test_loader_name in zip(test_loader_list, test_loader_names):
                    if (completed_steps+1) % checkpointing_steps == 0 or completed_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        # accelerator.save_state(output_dir)

                        model.eval()
                        samples_seen = 0
                        output_dicts = []
                        for step, batch in tqdm(enumerate(test_loader)):
                            with torch.no_grad():
                                outputs = model(**batch)
                            predictions = outputs.logits.argmax(dim=-1) #if not is_regression else outputs.logits.squeeze()

                            logits = outputs.logits.detach()
                            for j in range(logits.size(0)):
                                probs = logits[j]  #F.softmax(logits[j], -1)
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

                        eval_metric = metric.compute()
                        logger.info(f"epoch {epoch}: {eval_metric}")

                        if test_loader_name == 'eval':
                            accelerator.wait_for_everyone()
                            # unwrapped_model = accelerator.unwrap_model(model).model
                            accelerator.unwrap_model(model).model.save_pretrained(
                                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(output_dir)
                            

                        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}

                        if test_loader_name == 'val':
                            all_results_output_path = os.path.join(output_dir, f"all_results_val.json")
                        else:
                            all_results_output_path = os.path.join(output_dir, f"all_results.json")
                        if os.path.isfile(all_results_output_path):
                            os.remove(all_results_output_path)

                        with open(all_results_output_path, "w") as f:
                            json.dump(all_results, f)

                        if test_loader_name == 'val':
                            output_path = os.path.join(output_dir, f'eval_res_val.json')
                        else:
                            output_path = os.path.join(output_dir, f'eval_res.json')
                        print(f'writing outputs to \'{output_path}\'')

                        if os.path.isfile(output_path):
                            os.remove(output_path)

                        with open(output_path, 'w+') as f:
                            for i, output_dict in enumerate(output_dicts):
                                output_dict_str = json.dumps(output_dict)
                                f.write(f'{output_dict_str}\n')


                        del output_dicts, all_results, output_dict, eval_metric, logits, probs, label, predictions, references, outputs
        
            if completed_steps > args.max_train_steps:
                break
            
            model.train()
            outputs = model(**train_batch)
            y = train_batch['labels']
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, y)

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().cpu().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                parameters_to_clip = [p for group in optimizer.param_groups for p in group['params']]
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            del outputs, loss, y
    

if __name__ == "__main__":
    main()