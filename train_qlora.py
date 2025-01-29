#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script: ptm-predictor/train_qlora.py

Streams chunked PTM data from multiple Parquet shards for multi-class token classification.
Uses QLoRA in 4-bit precision for parameter-efficient fine-tuning.

Example: 
python ptm-predictor/train_qlora.py \
    --model_name tattabio/gLM2_650M \
    --chunk_parquet_dir chunks \
    --learning_rate 5e-4 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --weight_decay 0.01 \
    --lora_alpha 1 \
    --lora_dropout 0.2 \
    --r 8 \
    --seed 1337 \
    --project_name "ptm_prediction"
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from accelerate import Accelerator
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datetime import datetime
import wandb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)

# --------------------------------------------------
# Argparse
# --------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-class token classification with streaming data.")
    parser.add_argument("--model_name", type=str, default="tattabio/gLM2_650M",
                        help="Name or path of the base model to fine-tune.")
    parser.add_argument("--chunk_parquet_dir", type=str, default="chunks",
                        help="Directory containing chunked Parquet shards.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Per-device training batch size.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="ptm_site_prediction",
                        help="W&B project name.")
    return parser.parse_args()

# --------------------------------------------------
# Metric Computation
# --------------------------------------------------
def compute_metrics(pred):
    """Compute multi-class token classification metrics."""
    logits, labels = pred
    predictions = np.argmax(logits, axis=2)

    # Flatten ignoring special tokens with label=-100
    valid_mask = (labels != -100)
    pred_flat = predictions[valid_mask]
    labels_flat = labels[valid_mask]

    # Accuracy
    acc = accuracy_score(labels_flat, pred_flat)

    # Macro-averaged precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='macro', zero_division=0
    )

    # Multi-class AUC
    try:
        auc_val = roc_auc_score(labels_flat, pred_flat, average='macro', multi_class='ovr')
    except ValueError:
        # Possibly not all classes present
        auc_val = float('nan')

    # MCC
    mcc = matthews_corrcoef(labels_flat, pred_flat)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_val,
        "mcc": mcc,
    }

# --------------------------------------------------
# Custom Weighted Trainer
# --------------------------------------------------
class WeightedTrainer(Trainer):
    """
    Custom Trainer to handle class weighting. Must define class_weights externally.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

        # Flatten
        active_loss = inputs["attention_mask"].view(-1) == 1
        active_logits = logits.view(-1, model.config.num_labels)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        loss = loss_fct(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_arguments()

    # Init W&B
    wandb.init(project=args.project_name)
    wandb.config.update(vars(args))

    accelerator = Accelerator()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Build a streaming dataset from multiple Parquet shards
    #    We'll assume all shards are named chunks_XXX.parquet in chunk_parquet_dir.
    data_files = [os.path.join(args.chunk_parquet_dir, f) for f in os.listdir(args.chunk_parquet_dir)
                  if f.endswith(".parquet")]
    if not data_files:
        raise ValueError(f"No Parquet shards found in {args.chunk_parquet_dir}")

    raw_dataset = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",       # single split for now
        streaming=True       # enable streaming
    )

    # Optional: split streaming dataset into train and eval
    # One approach: use .take(N) for evaluation, .skip(N) for training
    # or a random subsample. 
    # For brevity, let's create two splits of the streaming dataset. 
    # Note: This is not a perfect "random" split.
    # Obviously this is sacralige for not splitting based on sequence similarity, but it's fine for testing purposes. 
    train_dataset = raw_dataset.take(50000)       # first 50k rows as "val"
    eval_dataset  = raw_dataset.skip(50000).take(10000)  # next 10k as "eval"

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 3. Tokenization + label alignment
    #    Each row has "chunked_seq" (string) and "chunked_labels" (list of integers).
    #    We'll treat each character as a "word" if your model uses single-letter tokens.
    def tokenize_and_align_labels(examples):
        # examples is a batch of streaming rows
        seq_list = list(examples["chunked_seq"])
        label_list = examples["chunked_labels"]

        # Convert each sequence from a string to a list of single characters for tokenization
        # because is_split_into_words=True requires a list of tokens
        tokenized_inputs = tokenizer(
            [list(seq) for seq in seq_list],
            is_split_into_words=True,
            return_tensors=None,  # Return as Python lists
            padding="max_length",  # or "longest"
            truncation=True,
            max_length=512  # or 1024 if consistent with chunk size
        )

        all_labels = []
        for i, word_ids in enumerate(tokenized_inputs.word_ids_batch):
            labels_i = label_list[i]
            # word_ids might map each subtoken to a "word index" or None for special tokens
            aligned_labels = []
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # ignore special tokens
                else:
                    aligned_labels.append(labels_i[word_idx])
            all_labels.append(aligned_labels)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    # Because we are streaming, we apply `.map(...)` with batched=True carefully.
    # We specify a batch_size. The output remains a generator.
    # Note that some transforms or random shuffles might not be fully supported in streaming mode.
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=256
    )
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=256
    )

    # 4. Compute class weights (optional)
    #    In a streaming scenario, it's tricky to compute class weights from the entire dataset.
    #    You might have to sample from the data or do it offline. 
    #    For demonstration, we'll just do uniform weighting or skip it.
    #    If you have precomputed class_weights, you can load them:
    # class_weights = torch.tensor([ ... ], dtype=torch.float32).to(accelerator.device)
    class_weights = None  # or define your array if known

    # 5. Load the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # We do not know how many classes you have in total, 
    # but let's assume it matches the largest label in your data + 1.
    # For real usage, define `num_labels` from your PTM_LABEL2ID or from an offline analysis.
    # e.g. if you have 6 classes, do `num_labels=6`.
    num_labels = 6

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["query","key","value","classifier"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    # 7. Accelerator
    model = accelerator.prepare(model)

    # 8. Training arguments
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    training_args = TrainingArguments(
        output_dir=f"{args.model_name.replace('/', '_')}_qlora_ptm_sites_{timestamp}",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=1,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=2,
        no_cuda=False,
        seed=args.seed,
        fp16=True,
        report_to='wandb',
        optim="paged_adamw_8bit"
    )

    # 9. DataCollator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    # 10. Trainer
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,    # streaming dataset
        eval_dataset=eval_dataset,      # streaming dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    # 11. Train
    trainer.train()

    # 12. Save final model
    final_model_path = f"best_model_{args.model_name.replace('/', '_')}_{timestamp}"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)


if __name__ == "__main__":
    main()