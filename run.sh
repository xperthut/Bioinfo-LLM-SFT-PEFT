#!/bin/bash
for i in {60..99}; do
  echo "Submitting job for cluster identity score: $i"
  result=$(echo "scale=2; $i / 100" | bc)
  sbatch run_cluster.slum random_25mers_10M.fasta output_Aug_26_$i.txt $result 5000 20000
done