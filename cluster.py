from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import time
import uuid
import platform
import asyncio

#Initialize the spark session with optimize configuration
spark = SparkSession.builder \
        .appName("dnaBERT2_Clustering") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.default.parallelism", "100") \
        .getOrCreate()

#print(spark)

# Load dnaBERT2 model and tokenizer
model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Generata synthetic DNA sequences and binary features
def generate_synthetic_data(n_sequences=10000000, seq_length=100, n_features=40):
    np.random.seed(42)
    bases = ['A', 'C', 'G', 'T']
    sequences = [''.join(np.random.choice(bases, seq_length)) for _ in range(n_sequences)]
    features = np.random.randint(0,2,size=(n_sequences, n_features))
    return spark.createDataFrame(pd.DataFrame({
            "sequence": sequences,
            **{f"feature_{i}":features[:, i] for i in range(n_features)}
        }))

# Function to extract dnaBERT2 embeddings
def get_dnabert2_embedding(sequence):
    max_seq_len=100
    if len(sequence)>max_seq_len:
        sequence = sequence[:max_seq_len]
    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=max_seq_len)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = output[0] #[1, sequence_length, 768]
    embedding = torch.mean(hidden_states[0], dim=0).numpy() # Mean pooling: [768]
    return embedding.tolist()

# UDF for Spark
from pyspark.sql.functions import udf
from pyspark.ml.functions import array_to_vector
from pyspark.sql.types import ArrayType, FloatType
dnabert2_udf = udf(get_dnabert2_embedding, ArrayType(FloatType()))

async def main():
    # Start time
    start_time = time.time()

    # Generate synthetic data
    df = generate_synthetic_data()

    # Extract DNABERT-2 embedding
    df_with_embeddings = df.withColumn("dnabert2_embedding", array_to_vector(dnabert2_udf("sequence")))

    # Combine DNABERT-2 embeddings with binary features
    feature_columns = [f"feature_{i}" for i in range(10)]+['dnabert2_embedding']

    assembler = VectorAssembler(
        inputCols = feature_columns,
        outputCol = "features",
        handleInvalid = "skip"
    )

    data = assembler.transform(df_with_embeddings)

    # Configure KMeans
    kmeans = KMeans(k=100, seed=42, maxIter=20, featuresCol="features")
    model=kmeans.fit(data)

    # Get cluster assignments
    predictions = model.transform(data)

    # Save results
    predictions.select("sequence", "features", "prediction") \
            .write.mode("overwrite").parquet("dna_bert2_clusters_parquet")

    # Calculate and print execution time
    execution_time = time.time()-start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Stop Spart session
    spark.stop()

# Run main function
if platform.system()=="Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__=="__main__":
        asyncio.run(main())