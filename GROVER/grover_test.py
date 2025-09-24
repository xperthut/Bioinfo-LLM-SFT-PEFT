import torch
import pandas as pd
import pytorch_lightning as L
from transformers import AutoTokenizer
import grover
import os
from time import time

torch.set_float32_matmul_precision('high')

MODEL = "PoetschLab/GROVER"
DATA_PATH = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data"
CHK_PT_PATH = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints"
BEST_CHK_PT = {
    'mpra_data': 'epoch=03-val_loss=0.485.ckpt',
    'cao2021': 'epoch=02-val_loss=0.395.ckpt', 
    'hek': 'grover_ft_hek-.epoch=01-val_loss=0.501.ckpt', 
    'hek_muscle': 'grover_ft_hek_muscle-.epoch=01-val_loss=0.547.ckpt', 
    'hek_muscle_pc3': 'grover_ft_hek_muscle_pc3-.epoch=01-val_loss=0.567.ckpt', 
    'hek_pc3': 'grover_ft_hek_pc3-.epoch=01-val_loss=0.564.ckpt', 
    'muscle': 'grover_ft_muscle-.epoch=02-val_loss=0.520.ckpt', 
    'muscle_pc3': 'epoch=02-val_loss=0.563.ckpt', 
    'pc3': 'grover_ft_pc3-.epoch=01-val_loss=0.495.ckpt'
}

if __name__ == "__main__":
    #grover_ckpt = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints/grover_mpra_data/checkpoints/epoch=03-val_loss=0.485.ckpt"
    #test_data_csv = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data/randomUTRs/random_25mers_10M.csv"
    #col = "grover_mpra_data"

    for p in ['mpra_data', 'cao2021', 'cao2021/ratios/hek', 'cao2021/ratios/hek_muscle', 
              'cao2021/ratios/hek_muscle_pc3', 'cao2021/ratios/hek_pc3', 'cao2021/ratios/muscle', 
              'cao2021/ratios/muscle_pc3', 'cao2021/ratios/pc3']:
        
        test_data_csv = os.path.join(DATA_PATH, p, 'test.csv')

        print(f'Loading test data from {test_data_csv}')

        grover_tokenizer = AutoTokenizer.from_pretrained(MODEL)

        grover_dl = grover.DNADataModule(test_data=test_data_csv, 
                                    batch_size=128,
                                    max_length=512, 
                                    tokenizer=grover_tokenizer
                                    )
        grover_dl.setup()
        grover_dl = grover_dl.test_dataloader()

        result = {}

        for m in ['mpra_data', 'cao2021', 'hek', 'hek_muscle', 'hek_muscle_pc3', 'hek_pc3', 'muscle', 'muscle_pc3', 'pc3']:
            grover_ckpt = os.path.join(CHK_PT_PATH, f"grover_{m}", 'checkpoints', BEST_CHK_PT[m])
            print(f"Best checkpoint for model [{m}] is [{grover_ckpt}]")

            grover_model = grover.GroverClassifier.load_from_checkpoint(grover_ckpt)
            grover_model.eval()

            trainer = L.Trainer(
                accelerator="auto",
                devices=1,
                precision=32
            )

            try:
                print(f"Starting prediction for data [{p}] using model [{m}]...")
                st = time()
                grover_preds = trainer.predict(grover_model, grover_dl)
                print(f"Prediction done.. concatenating...")
                res_proba = torch.cat(grover_preds, dim=0)
                #for pred in grover_preds:
                #    res.extend(pred.tolist())
                et = time()
                print(f"Prediction completed in {((et - st)/60):.2f} minutes")
                print(f"Total predictions: {len(res_proba)}")
                
                result[f"{m}"] = (res_proba >= 0.5).to(torch.float)
                result[f"{m}_proba"] = res_proba
                print()
            except Exception as e:
                print(f"Error during prediction: {e}")
                grover_preds = None

        if len(result) > 0:
            print(f"Saving results to disk for data [{p}]")
            tmp = pd.DataFrame(result)
            fn = f"results/GROVER_test_{p.split('/')[-1]}_results_sep.csv"
            tmp.to_csv(fn, index=False)
            print(f"Results saved to {fn}")
            print()
            print()