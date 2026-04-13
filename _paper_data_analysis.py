"""Plots for the companion paper were generated using this file."""
import os
import numpy as np
import re
from collections import defaultdict
from nn.notebooks.initialise_notebook import *
from plots_utils import plot_ensemble_energy_data_with_reference, save_plots, plot_energy_convergence
from io_handlers import read_blosc
from json_encoder import format_json_string

filename_dict = {
    "ca-30": ["fem_cubic_an_square_201_IGA2x2_9IP_PE_CA_AN-30_ND_T500__plot_energy", 0.006],
    "ca-50": ["fem_cubic_an_square_201_IGA2x2_9IP_PE_CA_AN-50_ND_T500__plot_energy", 0.006],
    "oa-30": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN-30_ND_T500__plot_energy", 0.015],
    "oa-50": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN-50_ND_T500__plot_energy", 0.015],
    "pf4": ["fem_square_201_IGA2x2_9IP_PE_NO_PF4_ND_T500__plot_energy", 0.006],
    "kick": ["kick", 0.035],
    "manav": ["fem_cubic_an_square_201_IGA2x2_9IP_PE_CA_AN-30_ND_T500__plot_energy", 0.015],
    "d_100": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy", 0.015],
    "d_400": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy", 0.015],
    "mesh_101": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy", 0.015],
    "mesh_301": ["fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy", 0.015]
}

for k, v in filename_dict.items():
    filename = f"_paper_data/nn_data/{k}_filtered"
    disp = v[1]
    ref_filename = f"_paper_data/ref_data/{v[0]}.dat" if v[0] is not None else None
    try:
        save_plots(*(filename, disp, ref_filename, (8,6)),filename="en_"+k, plot_function=plot_ensemble_energy_data_with_reference)
        print(f"plot success for {k}")
    except Exception as e:
        print("files not found")
        print(e)

convergence_groups = {
    "disp_conv": {
        "dirs": [
            ("_paper_data/nn_data/d_100_filtered", "100 steps"),
            ("_paper_data/nn_data/oa_ref_filtered", "200 steps"),
            ("_paper_data/nn_data/d_400_filtered", "400 steps"),
        ],
        "max_disp": 0.015,
        "ref": "_paper_data/ref_data/fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy.dat",
        "xlabel": "Displacement Increment Count"
    },
    "mesh_conv": {
        "dirs": [
            ("_paper_data/nn_data/mesh_101_filtered", "1/101"),
            ("_paper_data/nn_data/oa_ref_filtered", "1/201"),
            ("_paper_data/nn_data/mesh_301_filtered", "1/301"),
        ],
        "max_disp": 0.015,
        "ref": "_paper_data/ref_data/fem_ortho_an_square_201_IGA2x2_9IP_PE_OA_AN+30_ND_T500__plot_energy.dat",
        "xlabel": "Mesh Spacing, m"
    },
}

for group_name, cfg in convergence_groups.items():
    sample_file = next(pathlib.Path(cfg["dirs"][0][0]).glob("*__plot_energy.dat"))
    energy_keys = [k for k in read_blosc(sample_file).keys() if k != "x"]

    for energy_key in energy_keys:
        try:
            save_plots(
                cfg["dirs"], cfg["max_disp"], cfg["ref"], (8, 6), energy_key, cfg["xlabel"],
                filename=f"en_{group_name}_{energy_key}",
                plot_function=plot_energy_convergence,
            )
            print(f"plot success for {group_name} — {energy_key}")
        except Exception as e:
            print(f"cailed for {group_name} — {energy_key}: {e}")



def parse_training_times(parent_folder: str) -> dict:
    """
    Extracts training times from log files in subfolders, grouped by subfolder name.
 
    Args:
        parent_folder: Path to the parent directory containing subfolders of logs.
 
    Returns:
        A dict keyed by subfolder name, each containing:
            - 'times': list of raw training times (seconds)
            - 'mean': mean training time
            - 'std': standard deviation of training times
    """
    pattern = re.compile(r"Total training time:\s*([\d.]+)\s*seconds")
    times_by_folder = defaultdict(list)
 
    for folder_name in sorted(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
 
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", errors="ignore") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            times_by_folder[folder_name].append(float(match.group(1)))
            except OSError:
                continue
 
    result = {}
    for folder_name, times in times_by_folder.items():
        arr = np.array(times)/60
        result[folder_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
 
    return result

print(format_json_string(parse_training_times("_paper_data/logs")))


