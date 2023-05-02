import os, time, datetime
os.chdir('/zhome/33/6/147533/BENDR-XAI')

end_crop = 5.0
window_length = 4.0 
n_processes = -1
snr = 100.0
edf_dir = "/work1/s194260/tuh_eeg_random/"
parcellation_name = "HCPMMP1_combined"
save_dir = "/work1/s194260/"
proj = True
exp_name = "random"

now = datetime.datetime.now()
now_str = now.strftime("%H%M%S_%d%m%y")

for high, low in zip([1.0, 4.0, 8.0, 12.0, 30.0], [4.0, 8.0, 12.0, 30.0, 70.0]):
    name = f"TUH_{high}_{low}_{parcellation_name}_{now_str}"

    job = f"""#!/bin/sh
#BSUB -J {name}
#BSUB -q hpc
#BSUB -n 25
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 3:00
#BSUB -o logs/output_{name}_%J.out 
#BSUB -e logs/error_{name}_%J.err 
module load scipy/1.9.1-python-3.10.7
module load cuda/11.7 
source /work1/s194260/BENDR-ENV/bin/activate
python3.10 data_process/TUH_processer.py --high_pass {high} --low_pass {low} --window_length {window_length} --end_crop {end_crop} --n_processes {n_processes} --snr {snr} --edf_dir {edf_dir} --parcellation_name {parcellation_name} --save_dir {save_dir} --proj {proj} --name {exp_name}"""

    with open('temp_submit.sh', 'w') as file:
        file.write(job)

    os.system('bsub < temp_submit.sh')
    time.sleep(0.5)
    os.remove('temp_submit.sh')