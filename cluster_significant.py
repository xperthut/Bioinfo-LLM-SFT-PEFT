import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min
import timeit, time
import concurrent.futures
import os
from scipy.stats import wilcoxon, fligner, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.api import qqplot
from tqdm import tqdm
import argparse

def get_raw_And_aggregate_data(data_path, cfilename='output_Aug_26_66.txt', gfilename='gc_content_predictions.csv'):   
    df_tmp1 = pd.read_csv(os.path.join(data_path, cfilename), sep='\t', header=None)
    df_tmp1.columns = ['ClsID','SeqID','Score','Status']
    df_tmp1['seq_index'] = [int(x.split('|')[1]) for x in df_tmp1.SeqID.values]
    #print(df_tmp1.shape, df_tmp1.seq_index.min(), df_tmp1.seq_index.max())
    #display(df_tmp1.head())

    df_tmp2 = pd.read_csv(os.path.join(data_path, gfilename))
    df_tmp2.predictions = df_tmp2.predictions.astype(int)
    df_tmp2['seq_index'] = df_tmp2.index

    #print(df_tmp2.shape)
    #display(df_tmp2.head())

    data = df_tmp1.merge(df_tmp2, on=['seq_index'])
    del [df_tmp1, df_tmp2]
    data = data.loc[data.Status.isin(['C', 'M']), ['sequence', 'seq_index', 'predictions', 'gc_content', 'ClsID', 'Status', 'Score']]
    #print(data.shape)
    #data['char'] = [f">gi|{a}|{b}|{c:.2f}" for a,b,c in zip(data.seq_index, data.predictions, 100*data.gc_content)]
    #data[['sequence','predictions', 'gc_content', 'ClsID']].to_csv(os.path.join(data_path, "Cluster_for_pred_score_whole.csv"), index=False)
    #display(data.head())

    tmp = data.groupby('ClsID')['seq_index'].apply(lambda x: len(x)).reset_index()
    tmp = tmp[tmp.seq_index>=2]
    print(f"Total non-singular clusters={tmp.ClsID.nunique()}")
    print(f"Cluster {tmp.ClsID[tmp.seq_index==tmp.seq_index.max()].values[0]} has {tmp.seq_index.max()} members.")
    print(tmp.shape)
    #print(tmp.ClsID.sample(10))

    data = data[data.ClsID.isin(tmp.ClsID.unique())]
    del [tmp]
    print(f"Total non-singular clusters={data.ClsID.nunique()}")
    #display(data.head())

    df_agg = data.groupby('ClsID').agg({
        'predictions': ['mean', 'std', 'count']
    }).reset_index()
    df_agg.columns = df_agg.columns.map('_'.join).str.strip('|')
    df_agg.rename(columns={
        'predictions_count': 'Count',
        'ClsID_':'ClsID'
    }, inplace=True)
    #display(df_agg.head())

    return data, df_agg

def get_cluster_bands(df_agg, low, high, nmid=100):
    low_score_clusters = sorted(df_agg.ClsID[df_agg.predictions_mean<low].unique())
    high_score_clusters = sorted(df_agg.ClsID[df_agg.predictions_mean>high].unique())
    mid_score_clusters = sorted(df_agg.ClsID[np.logical_and(df_agg.predictions_mean>=low, df_agg.predictions_mean<=high)].unique())
    lc = len(low_score_clusters)
    hc = len(high_score_clusters)
    mc = len(mid_score_clusters)

    sl = set(low_score_clusters)
    sh = set(high_score_clusters)
    sm = set(np.random.choice(mid_score_clusters, size=nmid, replace=False))

    def print_val(a,b):
        print(a, b, a+b, (a+b)*(a+b-1)//2)

    print_val(lc, mc)
    print_val(mc, hc)
    print_val(lc, hc)

    return sl, sh, sm

def process_time(start: float, end: float, s=""):
    elapsed = end - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken{' for ['+s+']' if len(s)>0 else ''}: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

class SignificantClusters:
    def __init__(self, cluster_labels_A, cluster_labels_B, typeA='Low', typeB='High', pid=0):
        self.sA = set(cluster_labels_A)
        self.sB = set(cluster_labels_B)
        self.typeA = typeA
        self.typeB = typeB
        self.cluster_labels = sorted(list(self.sA.union(self.sB)))
        self.pid = pid

        print(f"[Proc-{self.pid}] Total clusters to process={len(self.cluster_labels)}")
        print(f"[Proc-{self.pid}] {self.typeA} scored clusters={len(self.sA)}")
        print(f"[Proc-{self.pid}] {self.typeB} scored clusters={len(self.sB)}")

        self.significant_inter_clusters = {
            'ttest_ind':[],
            'mannwhitneyu':[],
            'wilcoxon':[],
            'fligner':[]
        }

        self.significant_intra_clusters = {
            'ttest_ind':[],
            'mannwhitneyu':[],
            'wilcoxon':[],
            'fligner':[]
        }

    def get_sig_intra_clusters(self):
        return self.significant_intra_clusters
    
    def get_sig_inter_clusters(self):
        return self.significant_inter_clusters

    def get_members(self, data, i,j):
        score1 = data.predictions[data.ClsID==self.cluster_labels[i]].values
        score2 = data.predictions[data.ClsID==self.cluster_labels[j]].values
        return score1, score2

    def get_cluster_id_in_order(self, c1, c2):
        return (c1, c2) if c1<=c2 else (c2, c1)

    def hypotheis_test(self, score1, score2, c1, c2, fname):
        # Perform the test
        _, p_val = fname(score1, score2)

        # Adjust for multiple hypothesis testing
        p_val_adj = multipletests(p_val, method='fdr_bh')[1]

        if p_val_adj < 0.05:
            if (c1 in self.sA and c2 in self.sB) or (c1 in self.sB and c2 in self.sA):
                self.significant_inter_clusters[fname.__name__].append(self.get_cluster_id_in_order(c1, c2))
            else:
                self.significant_intra_clusters[fname.__name__].append(self.get_cluster_id_in_order(c1, c2))

    def find_significant_clusters(self, data):
        st = timeit.default_timer()
        n = len(self.cluster_labels)
        for i in range(n):
            for j in range(i+1, n):
                c1 = self.cluster_labels[i]
                c2 = self.cluster_labels[j]

                score1 = data.predictions[data.ClsID==c1].values
                score2 = data.predictions[data.ClsID==c2].values
                min_sample = min(len(score1), len(score2))

                try:
                    # Perform the t-test
                    self.hypotheis_test(score1, score2, c1, c2, ttest_ind)

                    # Perform the Mann-Whitney U test
                    self.hypotheis_test(score1, score2, c1, c2, mannwhitneyu)

                    # Perform the Wilcoxon Signed-Rank Test
                    self.hypotheis_test(np.random.choice(score1, size=min_sample), np.random.choice(score2, size=min_sample), c1, c2, wilcoxon)

                    # Perform the Fligner-Killeen Test
                    self.hypotheis_test(score1, score2, c1, c2, fligner)

                except Exception as e:
                    print(f"Error for ({c1}, {c2}). Error details: {e}")

        process_time(st, timeit.default_timer(), s="Finding significant clusters")

    def save_results(self, result_path='./', add_timestamp=True):
        try:
            print(f"[Proc-{self.pid}] Saving results to {result_path} ...")

            # Combined the significant cluster pairs those passes all tests
            setV = set(self.significant_inter_clusters['ttest_ind'])
            for k in self.significant_inter_clusters:
                if k in ['ttest_ind', 'wilcoxon']: continue
                setV = setV.intersection(set(self.significant_inter_clusters[k]))

            res = {
                f'{self.typeB}_avg_score':[],
                f'{self.typeA}_avg_score':[]
            }

            for h in self.sB:
                for l in self.sA:
                    if h>l: h,l = l,h
                    if (h,l) in setV: 
                        if h in self.sB:
                            res[f'{self.typeB}_avg_score'].append(h)
                            res[f'{self.typeA}_avg_score'].append(l)
                        else:
                            res[f'{self.typeB}_avg_score'].append(l)
                            res[f'{self.typeA}_avg_score'].append(h)

            fn = f"significant_clusters_{self.typeA}_{self.typeB}_avgscore{f'_{time.time()}' if add_timestamp else ''}.csv"
            df = pd.DataFrame(res)
            df.to_csv(os.path.join(result_path, fn), index=False)
            print(f"[Proc-{self.pid}] File savesd to {os.path.join(result_path, fn)}")
        except Exception as e:
            print(f"[Proc-{self.pid}] Error in saving results. Error details: {e}")

        #return (df[f'{self.typeB}_avg_score'].unique(), df[f'{self.typeA}_avg_score'].unique())

def multiprocess_helper(data, df_agg, args, s):
    pid = os.getpid()
    print(f"[Proc-{pid}] Started...")
    st = timeit.default_timer()

    # sl = cluster ids whose prediction score mean < low
    # sh = cluster ids whose prediction score mean > high
    # sm = random cluster ids whose prediction score mean is in between low and high
    sl, sh, sm = get_cluster_bands(df_agg, low=args.low, high=args.high, nmid=args.nmid)
    low_score_clusters, high_score_clusters, mid_score_clusters = list(sl), list(sh), list(sm)
    print(len(low_score_clusters), len(high_score_clusters), len(mid_score_clusters))

    if s==(1,0,1):
        sc = SignificantClusters(low_score_clusters, high_score_clusters)
    elif s==(1,1,0):
        sc = SignificantClusters(low_score_clusters, mid_score_clusters, typeB='Mid')
    elif s==(0,1,1):
        sc = SignificantClusters(mid_score_clusters, high_score_clusters, typeA='Mid')
    
    sc.find_significant_clusters(data)

    print(f"[Proc-{pid}] Calling save_results...")
    res = sc.save_results(result_path=args.resultpath)

    process_time(st, timeit.default_timer(), s=f"[Proc-{pid}] completed...")

    return res


def start_process(args):
    data, df_agg = get_raw_And_aggregate_data(args.datapath, args.clusterfile, args.genefile)

    resultpath = args.resultpath.replace('.','').replace('/','')
    if len(resultpath)>0:
        os.makedirs(args.resultpath, exist_ok=True)

    # Split the products into chunks
    num_processes = mp.cpu_count()-1
    print(f"Total process={num_processes}")

    t = [(1,0,1)]+[(0,1,1), (1,1,0)]*5

    print('Start the multi-processing...')
    st = timeit.default_timer()

    # Create a thread pool executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_processes, len(t))) as executor:
        # Run each chunk in a separate process and collect the results
        [executor.submit(multiprocess_helper, data, df_agg, args, s) for s in t]
        #results_B = [future.result()[0] for future in futures]
        #results_A = [future.result()[1] for future in futures]

    print('End the multi-processing...')
    process_time(st, timeit.default_timer(), "Complete multiprocess")
    print('All processing completed...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True, help="path to the data folder")
    parser.add_argument("--resultpath", type=str, required=True, help="path to store the results")
    parser.add_argument("--clusterfile", type=str, required=True, help="cluster file name")
    parser.add_argument("--genefile", type=str, required=True, help="gene file name")
    parser.add_argument("--low", type=float, required=False, default=17.0, help="Maximum mean prediction score for low band")
    parser.add_argument("--high", type=float, required=False, default=21.0, help="Minimum mean prediction score for high band")
    parser.add_argument("--nmid", type=int, required=False, default=100, help="Number of random clusters to select from mid band")
    
    args = parser.parse_args()

    print(f"Data path: {args.datapath}")
    print(f"Result path: {args.resultpath}")
    print(f"Cluster file: {args.clusterfile}")
    print(f"Gene file: {args.genefile}")
    print(f"Low band max mean prediction score: {args.low}")
    print(f"High band min mean prediction score: {args.high}")
    print(f"Number of random mid band clusters: {args.nmid}")

    start_process(args)


