#
# Run on Ampere with 80GB GPUs
#
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy
from torch import cuda
import pytorch_lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from multimolecule import RnaTokenizer, UtrLmForSequencePrediction

torch.set_float32_matmul_precision('high')

MODEL = 'multimolecule/utrlm-te_el'
MODEL_NAME = 'utrlm'

class DNADataSet(Dataset):
    
    def __init__(self, path, tokenizer, max_length):
        self.df_XY = pd.read_csv(path, delimiter=',', skiprows=1, header=None)
        if self.df_XY.shape[1] == 2:
            self.X = self.df_XY[0].values.flatten()  # Assuming first column is X
            self.Y = self.df_XY[1].values.flatten()  # Assuming second column is Y
        elif self.df_XY.shape[1] > 2:
            print("Input CSV must have exactly two columns: one for sequences and one for labels.")
            self.X = self.df_XY[1].values.flatten()  # Assuming first column is X
            self.Y = self.df_XY[2].values.flatten()  # Assuming second column is Y
        
        self.tokenizer = tokenizer
        
        # Add padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.max_length = max_length
      
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx] 
        encoding = self.tokenizer(
            x, 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'label': torch.tensor(y, dtype=torch.long)
        }


class DNADataModule(L.LightningDataModule):
    """
    DataModule that reads a full data set and randomly partitions it into a train/validation set.
    An independent test/hold-out data set may be passed as well. 
    """
    def __init__(self, tokenizer, max_length,
            train_data=None, val_data=None, test_data=None, batch_size=32, num_workers=10, pin_memory=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers  # Set number of workers for DataLoader
        self.pin_memory = pin_memory  # Set pin_memory to True for faster data transfer to GPU

    def setup(self, stage=str):
        """
        stage: fit, test, validate, predict
        """
        if self.train_data is not None:
            self._train_dataset = DNADataSet(self.train_data, self.tokenizer, self.max_length)
        
        if self.val_data is not None:
            self._val_dataset = DNADataSet(self.val_data, self.tokenizer, self.max_length)

        if self.test_data is not None:
            self._test_dataset = DNADataSet(self.test_data, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)


class UTRLMClassifier(L.LightningModule):

    def __init__(self, 
                 name="utrlm_binary_classifier",
                 checkpoint_dir = "./checkpoints/",
                 batch_size=32, 
                 learning_rate = 0.001,
                 prob_threshold=0.50):
        super().__init__()
        self.model_name = name
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.prob_threshold = prob_threshold
        self.loss = nn.BCEWithLogitsLoss()
        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.model = UtrLmForSequencePrediction.from_pretrained(MODEL)


    def forward(self, batch):
        output = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logit = output.logits
        return logit.view(-1)

    
    def training_step(self, batch, batch_idx):
        y = batch['label'].to(torch.float)
        logit = self.forward(batch)
        loss = self.loss(logit, y)
        acc = self.train_accuracy(logit, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', acc, on_epoch=True)
        return {'loss': loss, 'label': y}


    def validation_step(self, batch, batch_idx):
        y = batch['label'].to(torch.float)
        logit = self.forward(batch)
        loss = self.loss(logit, y)
        acc = self.valid_accuracy(logit, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', acc, on_epoch=True)
        return {'loss': loss, 'label': y}

    
    def test_step(self, batch, batch_idx):
        y = batch['label'].to(torch.float)
        logit = self.forward(batch)
        loss = self.loss(logit, y)
        acc = self.test_accuracy(logit, y)
        self.log('test_loss', loss)
        self.log('test_accurancy', acc, on_epoch=True)
        return {'loss': loss, 'label': y}


    def predict_step(self, batch, batch_idx):
        logits = self.forward(batch)
        probs = torch.sigmoid(logits)
        return probs #(probs >= self.prob_threshold).to(torch.float)

 
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


    def on_train_end(self):
        # save model params
        torch.save(self.state_dict(), f"{os.path.join(self.checkpoint_dir, self.name)}.pth")
        print("Model parameters saved")

class UTRLM():
    def get_device(self):
        if cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def train_model(self, train, valid, test, log_dir, log_name, model_name, batch_size, max_length):
        device = self.get_device()

        # instantiate data module
        tokenizer = RnaTokenizer.from_pretrained(MODEL)

        # if specifying a train, validation and test set
        dm = DNADataModule(
            tokenizer=tokenizer, 
            max_length=max_length,
            train_data=train,
            val_data=valid,
            test_data=test,
            batch_size=batch_size
        )

        model = UTRLMClassifier(batch_size=batch_size, name=model_name)
        #csvlogger = L.loggers.CSVLogger(save_dir=args.log_dir, name=args.log_name)
        tblogger = TensorBoardLogger(log_dir, name=log_name)

        checkpoint_callback = L.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k = 5,
            mode="min",
            dirpath=f"{log_dir}/{log_name}/checkpoints",
            filename=(model_name + "-.{epoch:02d}")
        )

        trainer = L.Trainer(
            accelerator="mps", #gpu 
            devices=1, 
            min_epochs=1, 
            max_epochs=30,
            precision=32, 
            logger=tblogger,
            callbacks=[checkpoint_callback
        ])
        
        trainer.fit(model, dm)
        
        # test
        trainer.test(ckpt_path="best", datamodule=dm)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dbname", type=str, required=True, help="Training CSV file")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--max-length", type=int, required=True, help="Max sequence-length")
    
    args = parser.parse_args()

    base_data_path = f'/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data/{args.dbname}'
    base_log_path = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints/"
    
    # instantiate data module
    tokenizer = RnaTokenizer.from_pretrained(MODEL)

    # if specifying a train, validation and test set
    dm = DNADataModule(
        tokenizer=tokenizer, 
        max_length=args.max_length,
        train_data=os.path.join(base_data_path, 'train.csv'),
        val_data=os.path.join(base_data_path, 'dev.csv'),
        test_data=os.path.join(base_data_path, 'test.csv'),
        batch_size=args.batch_size,
        num_workers=127,  # Adjust based on your system's capabilities
        pin_memory=True  # Set to True for faster data transfer to GPU
    )

    model = UTRLMClassifier(batch_size=args.batch_size, name=MODEL_NAME)
    #csvlogger = L.loggers.CSVLogger(save_dir=args.log_dir, name=args.log_name)
    tblogger = TensorBoardLogger(base_log_path, name=f"{MODEL_NAME}_{args.dbname.split('/')[-1]}")

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k = 5,
        mode="min",
        dirpath=os.path.join(base_log_path, f"{MODEL_NAME}_{args.dbname.split('/')[-1]}", "checkpoints"),
        filename=("{epoch:02d}-{val_loss:.3f}")
    )

    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        min_epochs=1, 
        max_epochs=30,
        precision=32, 
        logger=tblogger,
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, dm)
    
    # test
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
