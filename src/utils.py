import pickle
import json
import os
import torch

def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    fname_tmp = fname + "_tmp.pkl"
    with open(fname_tmp, "wb") as f:
        pickle.dump(data, f)
    os.rename(fname_tmp, fname)


def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def torch_save(fname, obj):
    """"Save data in torch format."""
    # Define names of temporal files
    fname_tmp = fname + ".tmp"

    torch.save(obj, fname_tmp)
    os.rename(fname_tmp, fname)

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines