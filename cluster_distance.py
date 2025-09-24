import pandas as pd
import numpy as np
import os
from sklearn.metrics import pairwise_distances_argmin_min
import timeit
import concurrent.futures
import multiprocessing as mp
from collections import Counter
import argparse
from tqdm import tqdm

class ClusterData:
    def __init__(self, data_folder, start, end, ):
        self.data_folder = data_folder
        self.data = None
        self.start = start
        self.end = end

    def get_data(self):
        return self.data
    
    def get_cluster_labels(self):
        if isinstance(self.data, pd.DataFrame):
            return sorted(self.data.ClsID.unique())
        return None

    def generate_cluster_data(self, cluster_file_name, gene_file_name, min_limit=2):
        """
            Parameters:
            =====================
                cluster_file_name (str): The file name where clustering information belongs, specially the output file after running MeShClust
                gene_file_name (str): The file name where gene name with other information belongs
                min_limit (int) [default=2]: The minimum number of member should belings at each cluster

            Return:
            =======================
            data (Pandas Dataframe): Dataframe contains cluster membership where each cluster has more than min_limit number of members
        """
        if min_limit<2: min_limit=2

        df_cluster = pd.read_csv(os.path.join(self.data_folder, cluster_file_name), sep='\t', header=None)
        df_cluster.columns = ['ClsID','SeqID','Score','Status']
        df_cluster['seq_index'] = [int(x.split('|')[1]) for x in df_cluster.SeqID.values]
        
        df_gene = pd.read_csv(os.path.join(self.data_folder, gene_file_name))
        df_gene.predictions = df_gene.predictions.astype(int)
        df_gene['seq_index'] = df_gene.index
        
        self.data = df_cluster.merge(df_gene, on=['seq_index'])
        del [df_cluster, df_gene]
        self.data = self.data.loc[self.data.Status.isin(['C', 'M']), ['sequence', 'seq_index', 'predictions', 'gc_content', 'ClsID', 'Status', 'Score']]
        print(f"Data size after merge={self.data.shape}")

        tmp = self.data.groupby('ClsID')['seq_index'].apply(lambda x: len(x)).reset_index()
        tmp = tmp[tmp.seq_index>min_limit]
        cls_lbl = sorted(tmp.ClsID.unique())
        print(f"Total clusters that has at least {min_limit} members = {len(cls_lbl)}")

        self.data = pd.DataFrame(self.data[self.data.ClsID.isin(cls_lbl[self.start:self.end])])
        del tmp
        print(f"Data size after filter={self.data.shape}")

def process_time(start: float, end: float, s=""):
    elapsed = end - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken{' for ['+s+']' if len(s)>0 else ''}: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def kmer_frequency(seq, k):
    """
    Calculate k-mer frequencies for a DNA sequence.

    Parameters:
    seq (str): DNA sequence
    k (int): k-mer size

    Returns:
    dict: k-mer frequencies
    """
    return Counter([seq[i:i+k] for i in range(len(seq)-k+1)])

def get_random_seq(df, clsIds, N=100):
    seq=[]
    cIDs=[]
    n=int(np.random.randint(low=50, high=101, size=1))
    
    for c in clsIds:
        seq.extend(df.sequence[df.ClsID==c].sample(n).values.tolist())
        cIDs.extend([c]*n)

    return seq, cIDs

def calculate_distances(df, cluster_IDs, k=2, rep=1000, sample_size=100):
    """
    Calculate inter-cluster and intra-cluster distances.

    Parameters:
    df: Dataset contains sequences and cluster ids
    cluster_IDs (list): unique cluster labels
    k (int): k-mer size
    rep (int): Repeat the process to compute distribution
    sample_size (int): Number of samples to pull from each cluster
    cluster_size (int): Number of clusters

    Returns:
    Intra_cluster_distance (list): list of intra cluster distance
    Inter_cluster_distance (list): list of inter cluster distance
    """

    #cluster_IDs = np.random.choice(df.ClsID.unique(), size=cluster_size)

    inter_CD=[]
    intra_CD=[]
    
    print(f"[Proc-{os.getpid()}] Started...")
    st = timeit.default_timer()

    for _ in tqdm(range(rep), desc=f"[Proc-{os.getpid()}]"):
        sequences, cluster_labels = get_random_seq(df, cluster_IDs, sample_size)
        print(f"[Proc-{os.getpid()}] Total clusters={len(cluster_labels)}, Total sequences={len(sequences)}.")

        # Calculate k-mer frequencies for each sequence
        freqs = [kmer_frequency(seq, k) for seq in sequences]

        # Convert frequencies to numerical representations
        num_freqs = []
        for freq in freqs:
            num_freq = [freq.get(kmer, 0) for kmer in set.union(*[set(f.keys()) for f in freqs])]
            num_freqs.append(num_freq)

        # Calculate inter-cluster distance
        inter_cluster_distance = 0
        for i in set(cluster_labels):
            cluster_freqs = [num_freqs[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
            cluster_center = np.mean(cluster_freqs, axis=0)
            inter_cluster_distance += np.sum(pairwise_distances_argmin_min(cluster_center.reshape(-1,1), np.array(cluster_freqs).reshape(-1,1))[1])

        inter_cluster_distance /= len(set(cluster_labels))
        inter_CD.append(inter_cluster_distance)

        # Calculate intra-cluster distance
        intra_cluster_distance = 0
        for i in set(cluster_labels):
            cluster_freqs = [num_freqs[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
            intra_cluster_distance += np.sum(pairwise_distances_argmin_min(np.mean(cluster_freqs, axis=0).reshape(-1,1), np.array(cluster_freqs).reshape(-1,1))[1])

        intra_cluster_distance /= len(sequences)
        intra_CD.append(intra_cluster_distance)
    
    process_time(st, timeit.default_timer(), s=f"Proc-{os.getpid()}")
    print(f"[Proc-{os.getpid()}] Finished...")
    return (inter_CD, intra_CD)

def multiprocess_handler(cd, start, end, cis):
    df = cd.get_data()
    cluster_labels = cd.get_cluster_labels()

    # Define the number of products
    num_products = len(cluster_labels)
    print(f'Total items={num_products}')

    # Split the products into chunks
    num_processes = mp.cpu_count()-1
    print(f"Total process={num_processes}")
    #d = np.arange(start, end, 1)
    np.random.shuffle(cluster_labels)
    chunks = np.array_split(cluster_labels, num_processes)
    print(f'Total chunks={len(chunks)}')
    for i, chunk in enumerate(chunks):
        print(f"Chunk-{i+1}={len(chunk)}")

    print('Start the multi-processing...')
    st = timeit.default_timer()

    # Create a thread pool executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Run each chunk in a separate process and collect the results
        futures = [executor.submit(calculate_distances, df, chunk) for chunk in chunks]
        results_inter = [future.result()[0] for future in futures]
        results_intra = [future.result()[1] for future in futures]

    print('End the multi-processing...')
    process_time(st, timeit.default_timer(), "Complete multiprocess")
    print('Combinning the results...')
    # Flatten the results into a single array
    results_inter = [item for sublist in results_inter for item in sublist]
    results_intra = [item for sublist in results_intra for item in sublist]

    # Print the results
    tmp = pd.DataFrame({
        'Intra':results_intra,
        'Inter':results_inter
    })
    os.makedirs('../results', exist_ok=True)
    tmp.to_csv(f'../results/Intra_inter_distance_{cis}_{start}_{end}.csv', index=False)
    print(f'Saved data to ../results/Intra_inter_distance_{cis}_{start}_{end}.csv')

def main(start, end, data_folder, cluster_file, gene_file, min_sample, cluster_identity_score):
    st = timeit.default_timer()
    cd = ClusterData(data_folder, start, end)
    cd.generate_cluster_data(cluster_file, gene_file, min_sample)
    process_time(st, timeit.default_timer(), "prepare data")
    
    multiprocess_handler(cd, start, end, cluster_identity_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index of the cluster labels")
    parser.add_argument("--end", type=int, required=True, help="End index (Exclude) of the cluster labels")
    parser.add_argument("--df", type=str, required=True, help="Location of data folder")
    parser.add_argument("--cf", type=str, required=True, help="Name of the file that contains clustering information")
    parser.add_argument("--gf", type=str, required=True, help="Name of the file that contains gene information")
    parser.add_argument("--ms", type=int, required=False, default=100, help="Minimum number of samples randomly pool")
    parser.add_argument("--cis", type=int, required=True, default=80, help="Cluster identity score")

    args = parser.parse_args()

    print(f"Start index={args.start}")
    print(f"End index={args.end}")
    print(f"Folder location of data={args.df}")
    print(f"Cluster file name={args.cf}")
    print(f"Genotype file name={args.gf}")
    print(f"Minimum membership of selected cluster={args.ms}")
    print(f"Cluster identity score={args.cis}")

    main(args.start, args.end, args.df, args.cf, args.gf, args.ms, args.cis)