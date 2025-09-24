import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_data(file_path):
    """
    Reads data from a CSV file and returns a DataFrame.
    
    :param file_path: str, path to the CSV file
    :return: pd.DataFrame
    """
    print(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    # Row wise sum
    return df.values.sum(axis=1).tolist()

def combine_result():
    df = []
    base = '/projects/wg-GenAI4Bio/for_Methun/5UTR_mRNA_project/mkamruz/'
    for f in ['dnaBERT2/Code/DNABERT2_results.csv',
              'UTRLM/Code/UTRLM_results.csv', 
              'GROVER/Code/GROVER_results.csv']:
        file = os.path.join(base, f)
        df.append(read_data(file))
        
    df=np.array(df).T
    print(df.shape)
    return df.sum(axis=1)

def plot_results():
    res = combine_result()
    print("Total rows=",len(res))

    sns.displot(res, kde=True, )
    plt.xlabel('Sum of predictions')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sum of Rows')
    plt.savefig('sum_distribution.png')
    plt.close()

if __name__ == "__main__":
    plot_results()
