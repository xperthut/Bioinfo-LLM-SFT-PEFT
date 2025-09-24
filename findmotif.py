import subprocess
import os

def call_slurm(sbatch_command):
    try:
        # Execute the sbatch command
        result = subprocess.run(sbatch_command, capture_output=True, text=True, check=True, shell=True)
        job_id = result.stdout.strip().split()[-1] # Assuming sbatch returns job ID on last line
        print(f"Job submitted successfully. Job ID: {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")

if __name__=="__main__":
    sub = 'v1'
    output_dir = f'/home/mkamruz/projects/5UTR/Data/motif_result/ALL_RANDOM_SHUFFLE_NOC/{sub}/STREAM'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.join('/home/mkamruz/projects/5UTR/Data', 'motif_all_random_noc', sub)
    #files = os.listdir(input_dir)
    #for fn in [f[:-3] for f in files]:
    
    i=1
    cmd = f"sbatch run_motif_band.slum {os.path.join(input_dir, 'cluster_L.fa')}  {os.path.join(output_dir, f'cluster_L_M_{i}')} 10 {os.path.join(input_dir, f'cluster_M_{i}.fa')} 8"
    print(cmd)
    call_slurm(cmd)

    i=2
    cmd = f"sbatch run_motif_band.slum {os.path.join(input_dir, 'cluster_H.fa')}  {os.path.join(output_dir, f'cluster_H_M_{i}')} 10 {os.path.join(input_dir, f'cluster_M_{i}.fa')} 8"
    print(cmd)
    call_slurm(cmd)
