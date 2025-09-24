import torch
import pandas as pd
import pytorch_lightning as L
from multimolecule import RnaTokenizer
import utrlm
import os
from time import time

torch.set_float32_matmul_precision('high')

MODEL = "multimolecule/utrlm-te_el"
DATA_PATH = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data"
CHK_PT_PATH = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints"
BEST_CHK_PT = {
    'mpra': 'utrlm_ft_mpra-.epoch=09-val_loss=0.479.ckpt',
    'cao2021': 'epoch=05-val_loss=0.459-v1.ckpt', 
    'hek': 'utrlm_hek-.epoch=14-val_loss=0.678.ckpt', 
    'hek_muscle': 'utrlm_ft_hek_muscle-.epoch=06-val_loss=0.681.ckpt', 
    'hek_muscle_pc3': 'utrlm_ft_hek_muscle_pc3-.epoch=13-val_loss=0.617.ckpt', 
    'hek_pc3': 'utrlm_hek_pc3-.epoch=11-val_loss=0.602.ckpt', 
    'muscle': 'utrlm_ft_muscle-.epoch=01-val_loss=0.569.ckpt', 
    'muscle_pc3': 'utrlm_ft_muscle_pc3-.epoch=13-val_loss=0.656.ckpt', 
    'pc3': 'utrlm_ft-.epoch=08-val_loss=0.560.ckpt'
}

if __name__ == "__main__":
    for p in ['mpra_data', 'cao2021', 'cao2021/ratios/hek', 'cao2021/ratios/hek_muscle', 
              'cao2021/ratios/hek_muscle_pc3', 'cao2021/ratios/hek_pc3', 'cao2021/ratios/muscle', 
              'cao2021/ratios/muscle_pc3', 'cao2021/ratios/pc3']:
        
        test_data_csv = os.path.join(DATA_PATH, p, 'test.csv')

        print(f'Loading test data from {test_data_csv}')

        utrlm_tokenizer = RnaTokenizer.from_pretrained(MODEL)

        utrlm_dl = utrlm.DNADataModule(test_data=test_data_csv, 
                                    batch_size=128,
                                    max_length=512, 
                                    tokenizer=utrlm_tokenizer,
                                    num_workers=127,
                                    pin_memory=True
                                    )
        utrlm_dl.setup()
        utrlm_dl = utrlm_dl.test_dataloader()

        result = {}

        # grover_cao2021         grover_muscle_pc3             utrlm_hek_pc3
        # grover_hek             grover_pc3                    utrlm_mpra
        # grover_hek_muscle      utrlm_cao2021                 utrlm_muscle
        # grover_hek_muscle_pc3  utrlm_hek                     utrlm_muscle_pc3
        # grover_hek_pc3         utrlm_hek_muscle              utrlm_pc3
        # grover_mpra_data       utrlm_hek_muscle_pc3
        # grover_muscle          utrlm_hek_muscle_pc3_len1024

        for m in ['mpra', 'cao2021', 'hek', 'hek_muscle', 'hek_muscle_pc3', 'hek_pc3', 'muscle', 'muscle_pc3', 'pc3']:

            utrlm_ckpt = os.path.join(CHK_PT_PATH, f"utrlm_{m}", 'checkpoints', BEST_CHK_PT[m])
            print(f"Best checkpoint for model [{m}] is [{utrlm_ckpt}]")

            #utrlm_ckpt = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/checkpoints/utrlm_cao2021/checkpoints/epoch=05-val_loss=0.459-v1.ckpt"
            #test_data_csv = "/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/DNABERT_data/randomUTRs/random_25mers_10M.csv"
            #col = "utrlm_cao2021"

            utrlm_model = utrlm.UTRLMClassifier.load_from_checkpoint(utrlm_ckpt)
            utrlm_model.eval()

            trainer = L.Trainer(
                accelerator="auto",
                devices=1,
                precision=32
            )

            try:
                print(f"Starting prediction for data [{p}] using model [{m}]...")
                st = time()
                utrlm_preds = trainer.predict(utrlm_model, utrlm_dl)
                res_proba = torch.cat(utrlm_preds, dim=0)
                #for pred in utrlm_preds:
                #    res.extend(pred.tolist())
                et = time()
                print(f"Prediction completed for data [{p}] using model [{m}] in {((et - st)/60):.2f} minutes")
                print(f"Total predictions: {len(res_proba)}")

                result[f"{m}"] = (res_proba >= 0.5).to(torch.float)
                result[f"{m}_proba"] = res_proba
                print()
            except Exception as e:
                print(f"Error during prediction for data [{p}] using model [{m}]: {e}")
                utrlm_preds = None

        if len(result) > 0:
            print(f"Saving results to disk for data [{p}]")
            tmp = pd.DataFrame(result)
            fn = f"results/UTRLM_test_{p.split('/')[-1]}_results_sep.csv"
            tmp.to_csv(fn, index=False)
            print(f"Results saved to {fn}")
            print()
            print()