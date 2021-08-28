"""Fine-tuning on the IMDB dataset"""

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed, EvalPrediction
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import torch

import transformers
transformers.logging.set_verbosity_debug()

import os
import sys
import logging

from scipy.special import softmax
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import dataclasses
from dataclasses import dataclass, field
from typing import Optional
# We use dataclass-based configuration objects, let's define the one related to 
# which model we are going to train here:
@dataclass
class AdditionalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


def preprocess_examples(tokenizer, examples):
    """Preprocess batched examples using a pre-trained BERT tokeniser."""
    tokenized_examples = tokenizer(examples['text'], truncation=True, padding=True)
    tokenized_examples["labels"] = [val for val in examples["label"]]
    return tokenized_examples


def load_dataset_for_fine_tuning(tokenizer, seed):
    """Load the IMDB dataset using datasets library.

    Notes:
        When using `datasets` and `Trainer`, we do not have to create a `DataLoader`.
        (i.e., do not wrap `datasets` with `DataLoader`, `Trainer` only accepts
        `torch.utils.data.dataset.Dataset`!)  
        Just pass the resulting data object to the instance of `Trainer`!
    
    References:
        https://huggingface.co/nlp/viewer/?dataset=imdb
        https://huggingface.co/datasets/imdb
        https://huggingface.co/docs/datasets/torch_tensorflow.html
    """
    dataset = load_dataset("imdb")
    train_set = dataset["train"]
    train_set.shuffle(seed=seed)
    test_set = dataset["test"]
    test_set.shuffle(seed=seed)

    print("Preprocess training samples...")
    train_set = train_set.map(lambda examples: preprocess_examples(tokenizer, examples), 
                              batched=True, batch_size=5000)
    print("Preprocess test samples...")
    test_set = test_set.map(lambda examples: preprocess_examples(tokenizer, examples), 
                            batched=True, batch_size=5000)

    train_set.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])
    test_set.set_format(type='torch', columns=['input_ids','attention_mask', 'labels'])

    return train_set, test_set


def main():
    ##########
    # Configs
    ##########
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((AdditionalArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    ##########
    # Setup logging
    ##########
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    ##########
    # Load pretrained model/data and tokenizer
    ##########
    config = AutoConfig.from_pretrained(args.model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config
    )

    ##########
    # Get datasets
    ##########
    # Do not pass the resulting data to `DataLoader`!
    train_set, test_set = load_dataset_for_fine_tuning(tokenizer, training_args.seed)

    ##########
    # Set up evaluation metrics
    ##########
    def compute_metrics_fn(p: EvalPrediction):
        """
        Compute accuracy, f1, precision, and recall.

        References:
            https://huggingface.co/transformers/main_classes/trainer.html#transformers.EvalPrediction
        """
        # preprocess
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = softmax(preds, axis=1)[:, 1] # proba
        preds = (preds >= 0.5).astype(int) # binary
        
        # compute each metric using scikit-learn
        acc = accuracy_score(p.label_ids, preds)
        f1 = f1_score(p.label_ids, preds)
        prec = precision_score(p.label_ids, preds)
        rec = recall_score(p.label_ids, preds)

        return {"acc": acc, "f1": f1, "precision": prec, "recall": rec}


    ##########
    # Set up a Trainer
    ##########
    tb_writer = SummaryWriter(training_args.logging_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tb_writer=tb_writer,
        compute_metrics=compute_metrics_fn
    )

    # training
    trainer.train()

    # final evaluation
    logger.info("*** Final evaluation ***")
    eval_result = trainer.evaluate(eval_dataset=test_set)
    for key, value in eval_result.items():
        logger.info("\t%s = %s", key, value)


if __name__ == "__main__":
    main()