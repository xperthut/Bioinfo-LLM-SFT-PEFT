import concurrent.futures
import numpy as np
import multiprocessing as mp
import argparse

def process_products(chunk):
    # Process the chunk
    return ([i*2 for i in chunk], [i*3 for i in chunk])

def main(start, end):
    # Define the number of products
    num_products = end-start+1
    print(f'Total items={num_products}')

    # Split the products into chunks
    num_processes = mp.cpu_count()
    print(f"Total process={num_processes}")
    d = np.arange(start, end, 1)
    np.random.shuffle(d)
    chunks = np.array_split(d, num_processes)
    print(f'Total chunks={len(chunks)}')
    for i, chunk in enumerate(chunks):
        print(f"Chunk-{i+1}={chunk}")

    # Create a thread pool executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Run each chunk in a separate process and collect the results
        futures = [executor.submit(process_products, chunk) for chunk in chunks]
        results1 = [future.result()[0] for future in futures]
        results2 = [future.result()[1] for future in futures]

    # Flatten the results into a single array
    results1 = [item for sublist in results1 for item in sublist]
    results2 = [item for sublist in results2 for item in sublist]

    # Print the results
    print(len(results1), len(results2))
    print(results1[:30])
    print(results2[:30])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index of the cluster labels")
    parser.add_argument("--end", type=int, required=True, help="End index (Exclude) of the cluster labels")
    parser.add_argument("--df", type=str, required=True, help="Location of data folder")
    parser.add_argument("--cf", type=str, required=True, help="Name of the file that contains clustering information")
    parser.add_argument("--gf", type=str, required=True, help="Name of the file that contains gene information")
    parser.add_argument("--ms", type=int, required=True, help="Minimum number of samples randomly pool")

    args = parser.parse_args()

    print(args.start)
    print(args.end)
    print(args.df)
    print(args.cf)
    print(args.gf)
    print(args.ms)

    main(args.start, args.end)
