import os
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    # GLUE
    "wnli": ("sentence1", "sentence2"), # 635
    "ax": ("premise", "hypothesis"),    # 1,459
    "rte": ("sentence1", "sentence2"),  # 2,490
    "mrpc": ("sentence1", "sentence2"), # 3,668
    "cola": ("sentence", None),         # 8,551
    "sst2": ("sentence", None),         # 67,349
    "qnli": ("question", "sentence"),   # 104,743
    "qqp": ("question1", "question2"),  # 363,846
    "mnli": ("premise", "hypothesis"),  # 392,702
    # SuperGLUE
    "cb": ("premise", "hypothesis"),    # 250
    # "axb": ("premise", "hypothesis"),    # 1,459
    # "axg": ("premise", "hypothesis"),    # 1,459
    "wic": ("sentence1", "sentence2"),  # 6,000
    "boolq": ("passage", "question"),   # 9,427
}



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='wnli',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
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
        default=256,
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
        default='roberta-base',
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
        default=32,
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
    parser.add_argument("--output_dir", type=str, default='/user/work/ad20999/outputs', help="Where to store the final model.")
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
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--laplace_hessian", type=str, default='kron')
    parser.add_argument("--laplace_sub", type=str, default='all')
    parser.add_argument("--laplace_prior", type=str, default='homo', help='homo, hetero')
    parser.add_argument("--laplace_optim_step", type=int, default=1000)
    parser.add_argument("--testing_set", type=str, default='val')
    parser.add_argument("--laplace_predict", type=str, default='mc_corr_100000', help='probit bridge bridge_norm mc_indep mc_corr')
    args = parser.parse_args()

    print(args)

    peft_method = 'lora'
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


def main(load_step):
    args = parse_args()
    args.load_step = load_step
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)
    
    peft_method = 'lora'
    laplace_output_dir = f'/user/work/ad20999/outputs_laplace/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}/step_{args.load_step}'
    
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
    if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
        raw_datasets = load_dataset("glue", args.task_name)
    elif args.task_name in ['cb', 'wic', 'boolq']:
        raw_datasets = load_dataset("super_glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    output_dir = args.output_dir + f'/step_{args.load_step}'

    peft_config = PeftConfig.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, ignore_mismatched_sizes=True, num_labels=num_labels)
    model = PeftModel.from_pretrained(model, output_dir)
    model.print_trainable_parameters()

    if args.laplace_sub == 'all':
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
    else:    
        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'out_proj' in name:
                param.requires_grad = True


    model.print_trainable_parameters()


    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
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
    if args.testing_set != 'val':
        val_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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
    model, optimizer, eval_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, eval_dataloader, eval_dataloader, lr_scheduler
    )

    model.eval()

    # Get the metric function
    if args.task_name is not None:
        if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
            metric = evaluate.load("glue", args.task_name, experiment_id=f"{laplace_output_dir}/la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")
        elif args.task_name in ['cb', 'wic', 'boolq']:
            metric = evaluate.load("super_glue", args.task_name, experiment_id=f"{laplace_output_dir}/la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}")
    else:
        metric = evaluate.load("accuracy")

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, **kwargs):
            output_dict = self.model(**kwargs)
            return output_dict['logits']

    model = WrappedModel(model)
    
    if args.laplace_prior == 'hetero':
        la = Laplace(model, 'classification',
                        subset_of_weights='all',  #args.laplace_sub,
                        hessian_structure=args.laplace_hessian)
    elif args.laplace_prior == 'homo':
        la = Laplace(model, 'classification', prior_precision=1.,
                        subset_of_weights='all',  #args.laplace_sub,
                        hessian_structure=args.laplace_hessian)
        

    print('----fitting Laplace-----')
    la.fit(train_dataloader)


    # if args.load_step == 9999:
    if args.laplace_optim_step > 0:
        if args.testing_set == 'val':
            prior_precision = la.optimize_prior_precision(method='marglik', n_steps=args.laplace_optim_step, lr=1e-1)
        elif args.testing_set != 'val':
            prior_precision = la.optimize_prior_precision(method='CV', val_loader=val_dataloader, log_prior_prec_min=-3, log_prior_prec_max=4, grid_size=100, lr=1e-1, n_steps=args.laplace_optim_step)
        print(f'prior precision: {prior_precision}')
        torch.save(prior_precision, f'{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')


    if args.save:
        with open(f'{laplace_output_dir}/la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pkl', 'wb') as f:
            dill.dump(la, f)

    
    samples_seen = 0
    output_dicts = []
    f_mu_list = []
    f_var_list = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # logits = la(batch)
            f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu_list.append(f_mu)
            f_var_list.append(f_var)

        samples = int(args.laplace_predict.split('_')[-1])
        f_mu = f_mu.expand(samples, -1, -1)
        f_var = f_var.expand(samples, -1, -1, -1)
        logits = torch.softmax(f_mu + (torch.linalg.cholesky(f_var).to(f_mu.dtype) @ torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(accelerator.device)).squeeze(-1), dim=-1).mean(0)

        predictions = logits.argmax(dim=-1)

        logits = logits.detach()
        for j in range(logits.size(0)):
            probs = logits[j]  #F.softmax(logits[j], -1)
            label = batch["labels"]
            output_dict = {
                'index': args.per_device_train_batch_size * step + j,
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
    torch.save(f_mu, f'{laplace_output_dir}/f_mu_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    torch.save(f_var, f'{laplace_output_dir}/f_var_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')


    output_path = os.path.join(output_dir, f'eval_res_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_predict}_{args.laplace_optim_step}.json')
    # output_path = os.path.join(output_dir, f'eval_res_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}.json')
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


    del model, train_dataloader, output_dicts, metric, la, f_mu, f_var, f_mu_list, f_var_list, eval_dataloader, val_dataloader
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    step_list = [0,*list(range(999, 10999, 1000))]
    for load_step in step_list:
        main(load_step)