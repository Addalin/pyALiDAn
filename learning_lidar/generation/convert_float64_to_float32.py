from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from learning_lidar.preprocessing import preprocessing as prep
from tqdm import tqdm
import numpy as np

# TODO - add this to utils , an as well to any "save" process of xarray dataset
def convert_to32(nc_path):
    ds = prep.load_dataset(str(nc_path))
    prep.save_dataset(ds, nc_path=str(nc_path))

if __name__ == '__main__':

    base_path = r"D:"
    paths = [r"\data_haifa\GENERATION",
             r"\data_haifa\DATA FROM TROPOS\molecular_dataset",
             r"\data_haifa\DATA FROM TROPOS\lidar_dataset"]

    exclude_paths = [r"D:\data_haifa\GENERATION\density_dataset\2017\04",
                     r"D:\data_haifa\GENERATION\density_dataset\2017\05"]

    exclude_files = []
    for exclude_path in exclude_paths:
        exclude_files.extend(list(Path(exclude_path).glob("**/*.nc")))
    exclude_files = [str(x) for x in exclude_files]
    exclude_files = set(exclude_files)

    for path in paths:
        full_paths = Path(base_path + path)
        file_list = set([str(pp) for pp in full_paths.glob("**/*.nc")])
        file_list = file_list - exclude_files
        print(f"found {len(file_list)} nc files in path {base_path + path}")
        for nc_path in tqdm(file_list):
            convert_to32(nc_path)

        # num_processes = min(cpu_count() - 1, len(file_list))
        # with Pool(num_processes) as p:
        #     for _ in tqdm(p.imap(convert_to32, file_list), total=len(file_list)):
        #         pass

            # p.map(convert_to32, file_list)
