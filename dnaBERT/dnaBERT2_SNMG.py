import sklearn.metrics
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
import timeit

import os
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import sklearn
import numpy as np

from peft import (
    LoraConfig,
    get_peft_model,
)

from tqdm import tqdm
import wandb

# Initialize W&B in offline mode
wandb.init(project="DNABERT2_mpra", entity="Nanobody", mode="offline")

#--------------------------------- DNA sequence related methods --------------------------------------------------------------

def get_alter_of_dna_sequence(sequence: str):
    """
    Get the reversed complement of the original DNA sequence.
    """
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """
    Generate k-mer string from DNA sequence.
    Transform a dna sequence to k-mer string
    """
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int, rank: int) -> List[str]:
    """
    Load or generate k-mer string for each DNA sequence.
    Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"[GPU{rank}] Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"[GPU{rank}] Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"[GPU{rank}] Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

#----------------------------------- Class to process supervised data --------------------------------------------------------------
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 gpu_id: int,
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1,
                 train_eval:bool=True,
                 global_rank: Optional[int] = None):

        super(SupervisedDataset, self).__init__()
        self.gpu_id = gpu_id
        if global_rank is None:
            self.global_rank = gpu_id
        else:
            self.global_rank = global_rank

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
            print(f"[GPU{self.global_rank}] Loaded {len(data)} samples from {data_path}...")
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning(f"[GPU{self.global_rank}] Perform single sequence classification...")
            print(f"[GPU{self.global_rank}] Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning(f"[GPU{self.global_rank}] Perform sequence-pair classification...")
            print(f"[GPU{self.global_rank}] Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            print(f"[GPU{self.global_rank}] Data format not supported.")
            raise ValueError(f"[GPU{self.global_rank}] Data format not supported.")
        
        if not train_eval:
            labels = [0]*len(labels)
        
        if kmer != -1:
            # only write file on the first process
            logging.warning(f"[GPU{self.global_rank}] Using {kmer}-mer as input...")
            print(f"[GPU{self.global_rank}] Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer, self.gpu_id)
        
        print(f"[GPU{self.global_rank}] Total text data={len(texts)}...")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        #print("Tokenization output:", output)  # Debugging line

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.unique_labels = list(set(labels))
        self.num_labels = len(self.unique_labels)

    def get_stats(self):
        print(f"[GPU{self.global_rank}] data size= {self.input_ids.shape}")
        print(f"[GPU{self.global_rank}] num_labels= {self.num_labels}")
        print(f"[GPU{self.global_rank}] unique_labels= {self.unique_labels}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print("Instances received by collator:", instances)  # Debugging line
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

#----------------------------------- Arguments for model, data and training --------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    train_file: Optional[str] = field(default="train.csv", metadata={"help": "training data file name."})
    eval_file: Optional[str] = field(default="dev.csv", metadata={"help": "validation data file name"})
    test_file: Optional[str] = field(default="test.csv", metadata={"help": "testing data file name"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: int = field(default=50)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    option: int = field(default=3)
    do_train: bool = field(default=True)

#----------------------------------- Helper functions --------------------------------------------------------------

def ddp_setup():
    print('Initialize the process')
    init_process_group(backend="nccl")

def ddp_cleanup():
    print('Terminating the process')
    destroy_process_group()

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    print("Saving the model snapshot")
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    """
    Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
    """
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    pred_proba = F.softmax(logits, dim=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    valid_pred_proba = pred_proba[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "auc": sklearn.metrics.roc_auc_score(
            valid_labels, valid_pred_proba, 
        )
    }


def compute_metrics(eval_pred):
    """
    Compute metrics used for huggingface trainer.
    """ 
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def prepare_dataloader(dataset: Dataset, batch_size: int, data_collator: DataCollatorForSupervisedDataset) -> DataLoader:
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True,
        sampler=DistributedSampler(dataset),
        collate_fn=data_collator,
    )

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Retrieve the latest checkpoint from the directory."""
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError("No checkpoints found in the directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return os.path.join(checkpoint_dir, latest_checkpoint)

def load_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    return tokenizer

def load_model(model_args: ModelArguments, training_args: TrainingArguments, num_labels: int):
    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
    )

    #print target model to identify modules
    print(model)

    # configure LoRA
    if model_args.use_lora:
        print('using lora')
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model

def load_trained_model(training_args: TrainingArguments):
    # load model
    saved_model_path = get_latest_checkpoint(training_args.output_dir)
    print(f"Loading model from {saved_model_path}")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(saved_model_path, 
        trust_remote_code=True)
    return model

def process_time(start: float, end: float):
    elapsed = end - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

#-------------------------------- Model trainer ---------------------------------------------

class Trainer:
    def __init__(self, gpu_id, model, train_data, eval_data, tokenizer, optimizer, training_args):
        self.gpu_id = gpu_id
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        self.training_args = training_args

        self.model = model.to(self.gpu_id)  # Move model to the appropriate GPU
        self.train_data = prepare_dataloader(train_data, 
                                             self.training_args.per_device_train_batch_size, 
                                             self.data_collator)
        
        self.eval_data = prepare_dataloader(eval_data,
                                             self.training_args.per_device_eval_batch_size, 
                                             self.data_collator)
        
        #self.optimizer = optimizer

        # Wrap the model in DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=training_args.find_unused_parameters)

        self.trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=self.train_data.dataset,
                                   eval_dataset=self.eval_data.dataset,
                                   data_collator=self.data_collator,
                                   optimizers=(optimizer, None)
                                   )
        
    def train(self):
        # Start training
        self.trainer.train()

        if self.training_args.save_model:
            self.trainer.save_state()
            safe_save_model_for_hf_trainer(self.trainer, self.training_args.output_dir)

        # Evaluate the model
        if self.training_args.eval_and_save_results:
            results = self.trainer.evaluate(eval_dataset=self.eval_data.dataset)
            results_path = os.path.join(self.training_args.output_dir, "results", self.training_args.run_name)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, "eval_results.json"), "w") as f:
                json.dump(results, f)

        print(f"[GPU{self.gpu_id}] Training completed.")

#--------------------------------- Model inference on test data --------------------------------------------------------------
class Inference:
    def __init__(self, gpu_id, model, test_data, tokenizer, training_args):
        self.gpu_id = gpu_id
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        self.training_args = training_args

        self.model = model.to(self.gpu_id)  # Move model to the appropriate GPU
        self.test_data = prepare_dataloader(test_data,
                                             self.training_args.per_device_eval_batch_size, 
                                             self.data_collator)
        
        
    def predict(self):
        print(f"[GPU{self.gpu_id}] Starting inference on test data...")

        self.model.eval()  # Set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for batch in tqdm(self.test_data, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.gpu_id)
                attention_mask = batch['attention_mask'].to(self.gpu_id)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds.tolist())

                # Append predictions and original sentences
                #for text, pred in zip(batch['texts'], preds):
                #    predictions.append((text, pred))

        print(f"[GPU{self.gpu_id}] Inference completed.")
        print(f"[GPU{self.gpu_id}] Total predictions: {len(predictions)}")
        print(f"[GPU{self.gpu_id}] Total positive: {sum(predictions)}")
        print(f"[GPU{self.gpu_id}] Total negative: {len(predictions)-sum(predictions)}")
#--------------------------------- Training function --------------------------------------------------------------

def train():

    local_rank = int(os.environ['LOCAL_RANK'])
    
    print(f"[GPU{local_rank}] Start parsing the arguments")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"[GPU{local_rank}] Parsing completed.")
    
    print(f"Output directory: {str(training_args.output_dir)}")
    os.makedirs(str(training_args.output_dir), exist_ok=True)

    # load tokenizer
    print('Loading tokenizer')
    tokenizer = load_tokenizer(model_args=model_args, training_args=training_args)
    
    # define datasets and data collator
    if training_args.do_train:
        print(f"[GPU{local_rank}] Creating supervised train dataset")
        train_dataset = SupervisedDataset(gpu_id=local_rank,
                                        tokenizer=tokenizer, 
                                        data_path=os.path.join(data_args.data_path, data_args.train_file), 
                                        kmer=data_args.kmer)
        
        print(f"[GPU{local_rank}] Training details:")
        train_dataset.get_stats()
        
        print(f"[GPU{local_rank}] Creating supervised validation dataset")
        val_dataset = SupervisedDataset(gpu_id=local_rank,
                                        tokenizer=tokenizer, 
                                        data_path=os.path.join(data_args.data_path, data_args.eval_file), 
                                        kmer=data_args.kmer)
        print(f"[GPU{local_rank}] validation details:")
        val_dataset.get_stats()

        # load model
        print(f'[GPU{local_rank}] Loading the model')
        model = load_model(model_args=model_args, training_args=training_args, num_labels=train_dataset.num_labels)

        print(f"[GPU{local_rank}] Initializing the Trainer class")
        trainer = Trainer(
            gpu_id=local_rank,
            model=model,
            train_data=train_dataset,
            eval_data=val_dataset,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate),
            training_args=training_args
        )

        print(f'[GPU{local_rank}] Starting the training process.')
        trainer.train()
        print(f'[GPU{local_rank}] Training process completed.')
        
    else:
        # Note: The test dataset is not used in training, but can be used for evaluation
        print(f"[GPU{local_rank}] Creating supervised test dataset")
        test_dataset = SupervisedDataset(gpu_id=local_rank,
                                          tokenizer=tokenizer, 
                                         data_path=os.path.join(data_args.data_path, data_args.test_file), 
                                         kmer=data_args.kmer, train_eval=False)
        
        print(f"[GPU{local_rank}] Test data details:")
        test_dataset.get_stats()

        # load model
        print(f'[GPU{local_rank}] Loading the model')
        model = load_trained_model(training_args=training_args)

        print(f"[GPU{local_rank}] Initializing the Inference class")
        inference = Inference(
            gpu_id=local_rank,
            model=model,
            test_data=test_dataset,
            tokenizer=tokenizer,
            training_args=training_args
        )

        print(f'[GPU{local_rank}] Starting the Inference process.')
        inference.predict()
        print(f'[GPU{local_rank}] Inference process completed.')

#--------------------------------- Main function --------------------------------------------------------------
def main():
    start = timeit.default_timer()

    # Initialize DDP
    ddp_setup()

    world_size = torch.cuda.device_count()  # Number of GPUs available
    if world_size < 1:
        raise ValueError("At least one GPU is required for DDP.")
    
    # Start training
    train()

    # Clean up DDP
    ddp_cleanup()
    
    # Clean up W&B run
    wandb.finish()

    end = timeit.default_timer()
    process_time(start, end)

if __name__ == "__main__":
    main()
