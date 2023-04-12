#!/bin/sh
#BSUB -J SNR_2
#BSUB -q hpc
#BSUB -n 30
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -o logs/output_SNR_%J.out 
#BSUB -e logs/error_SNR_%J.err 
module load scipy/1.9.1-python-3.10.7
module load cuda/11.7
source /zhome/33/6/147533/XAI/XAI-env/bin/activate
python3.10 notebooks/best_snr.py