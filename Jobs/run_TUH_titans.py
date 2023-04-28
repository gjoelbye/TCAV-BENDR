import os, time, datetime
os.chdir('/home/s194260/BENDR-XAI/')

end_crop = 5.0
window_length = 4.0 
n_processes = -1
snr = 100.0
edf_dir = "/scratch/s194260/tuh_eeg/"
parcellation_name = "HCPMMP1_combined"
save_dir = "/scratch/s194260/"

now = datetime.datetime.now()
now_str = now.strftime("%H%M%S_%d%m%y")


for high, low in zip([1.0, 4.0, 8.0, 12.0, 30.0], [4.0, 8.0, 12.0, 30.0, 70.0]):
    name = f"TUH_{high}_{low}_{parcellation_name}_{now_str}"
    
    job = f'''#!/bin/bash	

#SBATCH --job-name={name}
#SBATCH --output=logs/output_{name}_%J.out
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00
#SBATCH --mem=16gb

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

source ~/.bashrc
conda init bash
conda activate BENDR-XAI
python TUH_processer.py --high_pass {high} --low_pass {low} --window_length {window_length} --end_crop {end_crop} --n_processes {n_processes} --snr {snr} --edf_dir {edf_dir} --parcellation_name {parcellation_name} --save_dir {save_dir}

echo "Done: $(date +%F-%R:%S)"'''

    with open('temp_submit.sh', 'w') as file:
        file.write(job)

    os.system('sbatch temp_submit.sh')
    time.sleep(0.5)
    os.remove('temp_submit.sh')