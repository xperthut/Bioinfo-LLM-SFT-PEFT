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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pytorch_lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision('high')

MODEL = 'PoetschLab/GROVER'
MODEL_NAME = 'grover'

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
            train_data=None, val_data=None, test_data=None, batch_size=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

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
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=127, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=127, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=127, pin_memory=True)


# model
class GroverClassifier(L.LightningModule):

    def __init__(self, 
                 model_name = MODEL,
                 name="grover_binary_classifier",
                 checkpoint_dir = "./checkpoints/",
                 learning_rate = 0.0001,
                 prob_threshold = 0.50):
        super().__init__()
        self.model_name = model_name
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate = learning_rate
        self.prob_threshold = prob_threshold
        self.loss = nn.BCEWithLogitsLoss()
        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)


    def forward(self, batch):
        output = self.model(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'])
        # extract positive class logit
        pos_class_logits = output.logits[:, 1]
        return pos_class_logits


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


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dbname", type=str, required=False, help="Training CSV file")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--max-length", type=int, required=True, help="Max sequence-length")
    
    args = parser.parse_args()

    base_data_path = f'/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data/{args.dbname}'
    base_log_path = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints/"
    

    # instantiate data module
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # if specifying a specific train, validation and test set
    dm = DNADataModule(
        tokenizer=tokenizer, 
        max_length=args.max_length,
        train_data=os.path.join(base_data_path, 'train.csv'),
        val_data=os.path.join(base_data_path, 'dev.csv'),
        test_data=os.path.join(base_data_path, 'test.csv'),
        batch_size=args.batch_size
    )

    model = GroverClassifier(name=MODEL_NAME, learning_rate=1e-5)
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
