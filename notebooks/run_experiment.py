import os, time
os.chdir('/zhome/33/6/147533/XAI/BENDR-XAI')


for high, low in zip([1.0, 4.0, 8.0, 12.0, 30.0], [4.0, 8.0, 12.0, 30.0, 70.0]):
    name = f"activity_{high}_{low}_aparc"

    job = f"""#!/bin/sh
#BSUB -J {name}
#BSUB -q hpc
#BSUB -n 30
#BSUB -R "rusage[mem=10G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 2:00
#BSUB -o logs/output_{name}_%J.out 
#BSUB -e logs/error_{name}_%J.err 
module load scipy/1.9.1-python-3.10.7
module load cuda/11.7
source /zhome/33/6/147533/XAI/XAI-env/bin/activate
python3.10 notebooks/calculate_activity_parallel.py --high_pass {high} --low_pass {low}"""

    with open('temp_submit.sh', 'w') as file:
        file.write(job)

    os.system('bsub < temp_submit.sh')
    time.sleep(0.5)
    os.remove('temp_submit.sh')