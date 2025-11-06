import os
import scipy.io
from matplotlib import pyplot as plt

def load_all_annotations(hea_dir):
    """
    Load all .hea files in a directory and collect annotation labels.
    Returns: dict mapping filename -> list of labels, and set of unique labels
    """
    hea_files = [f for f in os.listdir(hea_dir) if f.endswith(".hea")]
    all_labels = []
    file_labels = {}

    for hea_file in hea_files:
        path = os.path.join(hea_dir, hea_file)
        with open(path, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                labels.append(line[1:].strip())  # remove '#' and whitespace
                all_labels.append(line[1:].strip())
        file_labels[hea_file] = labels

    unique_labels = set(all_labels)
    print("Unique annotation labels:", unique_labels)
    return file_labels, unique_labels

def load_ppg(mat_path, channel=2, key='val'):
    """
    Load PPG from .mat file, default channel=2 (PLETH), default key='val'
    """
    mat_data = scipy.io.loadmat(mat_path)
    ppg = mat_data[key][channel]
    return ppg

# Example usage
hea_dir = r"training"
file_labels, unique_labels = load_all_annotations(hea_dir)

# Optional: load and plot first PPG
first_file = list(file_labels.keys())[0].replace(".hea", ".mat")
ppg = load_ppg(os.path.join(hea_dir, first_file))
plt.plot(ppg[:3000])
plt.title("Channel 2 (PPG) sample")
plt.show()
