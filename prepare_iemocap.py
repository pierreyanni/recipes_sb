"""
Downloads and creates data manifest files for EMOCAP (https://sail.usc.edu/iemocap/).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the orginal training set intothree chunks.

Authors:
 * Mirco Ravanelli, 2021
"""

import os
import json
import shutil
import pathlib
import random
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

import gdown # for download from google drive

logger = logging.getLogger(__name__)
IEMOCAP_URL = "https://drive.google.com/uc?id=14-JdlRkJ6XTihMyp78AwjhfbdMLnPRPw"
SAMPLERATE = 16000


def prepare_data(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
    balanced=True
    ):
    """
    Prepares the json files for the IEMOCAP dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the transformed IEMOCAP dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respecively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    balanced : bool
        If True, train, valid and test sets balanced across emotions and
        speakers (so that same speakers present in all sets). If False, random
        assignment.

    Example
    -------
    >>> data_folder = '/path/to/emocap'
    >>> prepare_data(data_path, data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(
        data_folder, "IEMOCAP_ahsn_leave-two-speaker-out"
    )
    if not check_folders(train_folder):
        download_data(data_folder)

    # if not check_folders(train_folder):
    #     unpack_iemocap(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    
    # Random split the signal list into train, valid, and test sets.
    if balanced:
        wav_lists = get_all_files_balanced(train_folder)
        data_split = split_balanced_sets(wav_lists, split_ratio)
    else:
        wav_list = get_all_files(train_folder, match_and=extension)
        data_split = split_sets(wav_list, split_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list

    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-4:])

        # Getting emotion
        emo = path_parts[-2]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "emo": emo,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def get_all_files_balanced(train_folder):
    """Returns a list of lists of file names, one for each session and emotion.
    
    Arguments
    ---------
    train_folder: str
        Folder containing the data.
    
    Returns
    -------
    list containing lists of file names
    """
    wav_lists = []
    for dirpath, dirnames, filenames in os.walk(train_folder):
        if filenames and ".wav" in filenames[0]:
            files_paths= [os.path.join(dirpath, file) for file in filenames]
            wav_lists.append(files_paths)
    return wav_lists

def split_balanced_sets(wav_lists, split_ratio):
    """"Constructs a balanced data set by making sure that every pair 
    (session, emotion) is split between train, valid and test sets according
    to split ratio. Notice that for each session, there is one male and one female
    speaker.
    
    Arguments
    ---------
    wav_lists: list
        list of lists of all signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    Returns
    ------
    dictionary containing train, valid, and test splits.   
    """
    data_split = {k: [] for k in ['train', 'valid', 'test']}
    for wav_list in wav_lists:
        random.shuffle(wav_list)
        i_train = round(split_ratio[0] / sum(split_ratio) * len(wav_list))
        i_valid = round(sum(split_ratio[:2]) / sum(split_ratio) * len(wav_list))
        data_split['train'].extend(wav_list[:i_train])
        data_split['valid'].extend(wav_list[i_train:i_valid])
        data_split['test'].extend(wav_list[i_valid:])
    for split in data_split:
        random.shuffle(data_split[split])
    return data_split

def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarly respecting
    the split ratio within each class.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split

def download_data(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "IEMOCAP_processed.tar.gz")
    # download_file(IEMOCAP_URL, train_archive)
    dest_dir = pathlib.Path(train_archive).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(IEMOCAP_URL, train_archive, quiet=False) 
    shutil.unpack_archive(train_archive, destination)

def unpack_iemocap(destination):
    """unpacks file.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    shutil.unpack_archive(IEMOCAP_PATH, destination)
