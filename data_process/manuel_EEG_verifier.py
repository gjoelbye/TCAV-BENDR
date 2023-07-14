import mne
import matplotlib.pyplot as plt
import os
if os.getcwd().split("/")[-1] != 'BENDR-XAI': os.chdir("../")


selected = []
with open('data_process/tuh_final_selected.txt', 'r') as f:
    selected.extend(f.read().splitlines())

deselected = []
with open('data_process/tuh_final_deselected.txt', 'r') as f:
    deselected.extend(f.read().splitlines())

files = []
files = ['data/tuh_eeg/' + file for file in os.listdir('data/tuh_eeg/')]
files = [file for file in files if file not in selected and file not in deselected]

# Function to handle user input
def on_key(event):
    if event.key == 'y':
        with open('tuh_final_selected.txt', 'a') as f:
            f.write(current_file + '\n')
        
        plt.close()
    elif event.key == 'n':
        with open('tuh_final_deselected.txt', 'a') as f:
            f.write(current_file + '\n')
        
        plt.close()

# Iterate through the EDF files
for file_path in files:
    
    if file_path in selected or file_path in deselected:
        continue   
    
    # Load the EDF file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.resample(256, verbose=False)
    raw.filter(1, 70, fir_design='firwin', verbose=False)
    raw.notch_filter(60, fir_design='firwin', verbose=False)
    
    # Plot the data
    fig = raw.plot(show=False, verbose=False, scalings='auto')
    current_file = file_path
    plt.title(f'File: {file_path}\nPress "y" to save or any other key to skip')
    
    # Register the event handler and show the plot
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()