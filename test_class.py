from eeg_loader import EEGLoader

eeg = EEGLoader("", mode="mat")
eeg.write("out.edf")
