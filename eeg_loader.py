from os.path import splitext, extsep
from pathlib import Path
from typing import Literal
from edfio import EdfSignal, EdfAnnotation, Edf
from mne.io import read_raw_edf, read_raw_cnt, read_raw_brainvision
from pandas import DataFrame
from scipy.io import loadmat
from numpy import ndarray, fromfile, float32, str_


def ext_replace(path: Path, ext: str) -> Path:
    return Path(splitext(path)[0] + extsep + ext)


class EEGLoader:
    freq: int
    points: int
    data: list[EdfSignal] = list()
    annotations: list[EdfAnnotation] = list()

    def __init__(self, filename: Path, mode: Literal["auto", "cnt", "eeg", "mat", "set", "edf"]):
        match mode:
            case "auto":
                without_ext = splitext(filename)[0]
                match splitext(filename)[1][1:]:
                    case "eeg" | "vhdr" | "vmrk":
                        eeg = EEGLoader(without_ext + extsep + "vhdr", "eeg")
                    case "edf" | "mat" | "set" | "cnt":
                        eeg = EEGLoader(filename, splitext(filename)[1][1:])
                    case _:
                        raise NotImplementedError
                self.freq = eeg.freq
                self.points = eeg.points
                self.data = eeg.data
                self.annotations = eeg.annotations

            case "cnt":
                cnt = read_raw_cnt(input_fname=filename, preload=True)
                df: DataFrame = cnt.to_data_frame(scalings={"eeg": 1.0})
                for annot in cnt._annotations:
                    self.annotations.append(EdfAnnotation(onset=annot["onset"], text=annot["description"],
                                                          duration=None if annot["duration"] == 0.0 else annot[
                                                              "duration"]))
                for ch_name in cnt.ch_names:
                    self.data.append(
                        EdfSignal(data=df[ch_name].to_numpy(), sampling_frequency=cnt.info["sfreq"], label=ch_name))
                self.freq = cnt.info["sfreq"]
                self.points = cnt.n_times
            case "eeg":
                vhdr = read_raw_brainvision(filename, preload=True)
                vhdr.get_montage()
                df: DataFrame = vhdr.to_data_frame(scalings={"eeg": 1.0})
                for annot in vhdr.annotations:
                    self.annotations.append(EdfAnnotation(onset=annot["onset"], text=annot["description"],
                                                          duration=None if annot["duration"] == 0.0 else annot[
                                                              "duration"]))
                for ch_name in vhdr.ch_names:
                    self.data.append(
                        EdfSignal(data=df[ch_name].to_numpy(), sampling_frequency=vhdr.info["sfreq"], label=ch_name))
                self.freq = vhdr.info["sfreq"]
                self.points = vhdr.n_times

            case "mat" | "set":
                print(filename)
                mat: dict = loadmat(filename.__str__())
                # header: bytes
                # version: float
                # _globals: list[str]
                content: ndarray
                content = mat["EEG"]


                chanlocs = list(map(lambda x: x.item(), content["chanlocs"][0, 0]["labels"].flatten()))

                if type(content["data"][0, 0][0]) == str_:
                    print(content["data"][0, 0][0].item())

                    with ext_replace(filename, "fdt").open(mode="rb") as f:
                        data = fromfile(f, dtype=float32).reshape(
                            [content["nbchan"][0, 0].item(), content["pnts"][0, 0].item(), -1], order="F")
                else:
                    data = content["data"][0, 0]
                for pos, label in enumerate(chanlocs):
                    self.data.append(
                        EdfSignal(data=data[pos].T.flatten(),
                                  sampling_frequency=content["srate"][0, 0].item(), label=label))

                for _type, latency in content["event"][0, 0][["type", "latency"]][0]:
                    print(_type, latency)
                    print(_type.item())
                    print(type(_type.item()))
                    self.annotations.append(
                        EdfAnnotation(onset=latency.item() / content["srate"][0, 0].item(), text=str(_type.item()),
                                      duration=None))
                self.freq = content["srate"][0, 0].item()
                self.points = content["pnts"][0, 0].item() * content["trials"][0, 0].item()

            case "edf":
                edf = read_raw_edf(filename, preload=True)
                df: DataFrame = edf.to_data_frame(scalings={"eeg": 1.0})
                for annot in edf.annotations:
                    self.annotations.append(EdfAnnotation(onset=annot["onset"], text=annot["description"],
                                                          duration=None if annot["duration"] == 0.0 else annot[
                                                              "duration"]))
                for ch_name in edf.ch_names:
                    self.data.append(
                        EdfSignal(data=df[ch_name].to_numpy(), sampling_frequency=edf.info["sfreq"], label=ch_name,
                                  physical_range=(df[ch_name].min(), df[ch_name].max())))
                self.freq = edf.info["sfreq"]
                self.points = edf.n_times
            case _:
                raise NotImplementedError

        pass

    def write(self, path: Path):
        with Path(path).open("wb") as f:
            Edf(signals=self.data, annotations=self.annotations, data_record_duration=self.points / self.freq).write(f)
