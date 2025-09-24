import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
import pandas as pd

from peft import (
    LoraConfig,
    get_peft_model,
)

from tqdm import tqdm

import wandb

# Initialize W&B in offline mode
wandb.init(project="DNABERT2_test", entity="Nanobody", mode="offline")

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
            
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning(f"[GPU{self.global_rank}] Perform sequence-pair classification...")
            print(f"[GPU{self.global_rank}] Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            
        else:
            print(f"[GPU{self.global_rank}] Data format not supported.")
            raise ValueError(f"[GPU{self.global_rank}] Data format not supported.")
        
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

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.texts = texts

    def get_stats(self):
        print(f"[GPU{self.global_rank}] data size= {self.input_ids.shape}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, str]:
        return dict(input_ids=self.input_ids[i], texts=self.texts[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, str]:
        #print("Instances received by collator:", instances)  # Debugging line
        input_ids, texts = tuple([instance[key] for instance in instances] for key in ("input_ids", 'texts'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        #labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            texts=texts,
        )

#----------------------------------- Helper functions --------------------------------------------------------------

def ddp_setup():
    print('Initialize the process')
    init_process_group(backend="nccl")

def ddp_cleanup():
    print('Terminating the process')
    destroy_process_group()

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

def load_trained_model(output_dir: str):
    # load model
    saved_model_path = get_latest_checkpoint(output_dir)
    print(f"Loading model from {saved_model_path}")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(saved_model_path, trust_remote_code=True)
    return model

def process_time(start: float, end: float):
    elapsed = end - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


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
        pred_proba = []

        with torch.no_grad():
            for batch in tqdm(self.test_data, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.gpu_id)
                attention_mask = batch['attention_mask'].to(self.gpu_id)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds.tolist())

                proba = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                pred_proba.extend(proba[:,1].tolist())

                # Append predictions and original sentences
                #for text, pred in zip(batch['texts'], preds):
                #    predictions.append((text, pred))

        print(f"[GPU{self.gpu_id}] Inference completed.")
        return (predictions, pred_proba)

        
#--------------------------------- Prediction on 10M random UTR5' data function --------------------------------------------------------------

def predict():

    local_rank = int(os.environ['LOCAL_RANK'])
    
    print(f"[GPU{local_rank}] Start parsing the arguments")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"[GPU{local_rank}] Parsing completed.")

    # load tokenizer
    print('Loading tokenizer')
    tokenizer = load_tokenizer(model_args=model_args, training_args=training_args)
    
    # Note: The test dataset is not used in training, but can be used for evaluation
    print(f"[GPU{local_rank}] Creating supervised test dataset")
    test_dataset = SupervisedDataset(gpu_id=local_rank,
                                        tokenizer=tokenizer, 
                                        data_path=os.path.join(data_args.data_path, data_args.test_file), 
                                        kmer=data_args.kmer)
    
    print(f"[GPU{local_rank}] Test data details:")
    test_dataset.get_stats()
    
    result = {}
    for p in ['mpra_data', 'cao2021', 'cao2021/ratios/hek', 'cao2021/ratios/hek_muscle', 
              'cao2021/ratios/hek_muscle_pc3', 'cao2021/ratios/hek_pc3', 'cao2021/ratios/muscle', 
              'cao2021/ratios/muscle_pc3', 'cao2021/ratios/pc3']:
        
        chkpt_path = f"/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/mkamruz/dnaBERT2/eval/{p}/SNMG"
        if os.path.exists(chkpt_path):
            print(f"[GPU{local_rank}] Loading checkpoint from {chkpt_path}")
            print(f'[GPU{local_rank}] Loading the model')
            model = load_trained_model(chkpt_path)

            print(f"[GPU{local_rank}] Initializing the Inference class")
            inference = Inference(
                gpu_id=local_rank,
                model=model,
                test_data=test_dataset,
                tokenizer=tokenizer,
                training_args=training_args
            )

            print(f'[GPU{local_rank}] Starting the Inference process.')
            pred, proba = inference.predict()
            result[f"DNABERT2_{p.split('/')[-1]}"] = pred
            result[f"DNABERT2_{p.split('/')[-1]}_proba"] = proba
            #predictions = inference.predict()
            #output_df = pd.DataFrame(predictions, columns=['seq', 'label'])
            #output_df.to_csv('random5UTR_10M_predictions.csv', index=False)
            print(f'[GPU{local_rank}] Inference process completed.')

            del [model, inference]
        else:
            print(f"[GPU{local_rank}] Checkpoint {chkpt_path} does not exist, skipping...")
            

    if len(result) > 0:
        print(f"[GPU{local_rank}] Saving results to disk")
        tmp = pd.DataFrame(result)
        tmp.to_csv(f"DNABERT2_results_sept.csv", index=False)
        print(f"[GPU{local_rank}] Results saved to DNABERT2_results_sept.csv")

#--------------------------------- Prediction on test data function --------------------------------------------------------------

def test():

    local_rank = int(os.environ['LOCAL_RANK'])
    
    print(f"[GPU{local_rank}] Start parsing the arguments")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"[GPU{local_rank}] Parsing completed.")

    # load tokenizer
    print('Loading tokenizer')
    tokenizer = load_tokenizer(model_args=model_args, training_args=training_args)
    
    for p in ['mpra_data', 'cao2021', 'cao2021/ratios/hek', 'cao2021/ratios/hek_muscle', 
              'cao2021/ratios/hek_muscle_pc3', 'cao2021/ratios/hek_pc3', 'cao2021/ratios/muscle', 
              'cao2021/ratios/muscle_pc3', 'cao2021/ratios/pc3']:
        
        # Note: The test dataset is not used in training, but can be used for evaluation
        print(f"[GPU{local_rank}] Creating supervised test dataset for {p}")
        tfn = os.path.join(data_args.data_path, p, data_args.test_file)
        
        test_dataset = SupervisedDataset(gpu_id=local_rank,
                                            tokenizer=tokenizer, 
                                            data_path=tfn, 
                                            kmer=data_args.kmer)
        
        print(f"[GPU{local_rank}] Test data details [{tfn}]:")
        test_dataset.get_stats()

        result = {}

        for m in ['mpra_data', 'cao2021', 'cao2021/ratios/hek', 'cao2021/ratios/hek_muscle', 
            'cao2021/ratios/hek_muscle_pc3', 'cao2021/ratios/hek_pc3', 'cao2021/ratios/muscle', 
            'cao2021/ratios/muscle_pc3', 'cao2021/ratios/pc3']:
        
            chkpt_path = f"/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/mkamruz/dnaBERT2/eval/{m}/SNMG"
            if os.path.exists(chkpt_path):
                print(f"[GPU{local_rank}] Loading checkpoint from {chkpt_path}")
                print(f'[GPU{local_rank}] Loading the model')
                model = load_trained_model(chkpt_path)

                print(f"[GPU{local_rank}] Initializing the Inference class")
                inference = Inference(
                    gpu_id=local_rank,
                    model=model,
                    test_data=test_dataset,
                    tokenizer=tokenizer,
                    training_args=training_args
                )

                print(f'[GPU{local_rank}] Starting the Inference process.')
                pred, proba = inference.predict()
                result[f"{m.split('/')[-1]}"] = pred
                result[f"{m.split('/')[-1]}_proba"] = proba
                #predictions = inference.predict()
                #output_df = pd.DataFrame(predictions, columns=['seq', 'label'])
                #output_df.to_csv('random5UTR_10M_predictions.csv', index=False)
                print(f'[GPU{local_rank}] Inference process completed for model [{m}] on dataset [{p}].')
                print()
                del [model, inference]
            else:
                print(f"[GPU{local_rank}] Checkpoint {chkpt_path} does not exist, skipping...")

        if len(result) > 0:
            print(f"[GPU{local_rank}] Saving results to disk")
            tmp = pd.DataFrame(result)
            fn = f"results/DNABERT2_test_{p.split('/')[-1]}_results_sept.csv"
            tmp.to_csv(fn, index=False)
            print(f"[GPU{local_rank}] Results saved to {fn}")
            print()
            print()
            
    

#--------------------------------- Main function --------------------------------------------------------------
def main():
    start = timeit.default_timer()

    # Initialize DDP
    ddp_setup()

    world_size = torch.cuda.device_count()  # Number of GPUs available
    if world_size < 1:
        raise ValueError("At least one GPU is required for DDP.")
    
    # Start training
    #predict()

    # Start prediction on test data
    test()

    # Clean up DDP
    ddp_cleanup()
    
    # Clean up W&B run
    wandb.finish()

    end = timeit.default_timer()
    process_time(start, end)

if __name__ == "__main__":
    main()
