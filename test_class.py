from eeg_loader import EEGLoader

eeg = EEGLoader("", mode="mat") # filename here
eeg.write("out.edf")
