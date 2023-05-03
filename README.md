[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7424229.svg)](https://doi.org/10.5281/zenodo.7424229)

# pyALiDAn
## Python implementation of the **Atmospheric Lidar Data Augmentation (ALiDAn)** framework & a learning pytorch-based pipeline of lidar analysis.
ALiDAn is an end-to-end physics- and statistics-based simulation framework of lidar measurements [1]. This framework aims to promote the study of dynamic phenomena from lidar measurements and set new benchmarks. 

The repository also includes a spatiotemporal and synergistic lidar calibration approach [2], which forms a learning pipeline for additional algorithms such as inversion of aerosols, aerosol typing etc.

> Note:
This repository is still under final preparations. 
It will hold the supplemental data and code for the papers [1] and [2].
> To receive a notification when the code is ready, you are welcome to add our repository to your "star" & "watch" repositories :)


### References:

[1] Adi Vainiger, Omer Shubi, Yoav Schechner, Zhenping Yin, Holger Baars, Birgit Heese, Dietrich Althausen, "ALiDAn: Spatiotemporal and Multi--Wavelength Atmospheric Lidar Data Augmentation”, under review, 2022.

[2] Adi Vainiger, Omer Shubi, Yoav Schechner, Zhenping Yin, Holger Baars, Birgit Heese, Dietrich Althausen, "Supervised learning calibration of an atmospheric lidar” IEEE International Geoscience and Remote Sensing Symposium (2022).

### Acknowledgements:
I. Czerninski, Y. Sde Chen, M. Tzabari, Y.Bertschy , M. Fisher, J. Hofer, A. Floutsi, R. Hengst, I. Talmon, and D. Yagodin, The Taub Foundation, Ollendorff Minerva Center.
The authors acknowledge the financial contributions and the inspiring framework of the ERC Synergy Grant “CloudCT” (Number 810370).


pyALiDAn derives data from measurements, reanalyses, and assimilation databases such as [PollyNet](https://github.com/PollyNET/Pollynet_Processing_Chain), [AERONET](https://aeronet.gsfc.nasa.gov/new_web/data.html) by NASA , [GDAS](https://www.ncei.noaa.gov/products/weather-climate-models/global-data-assimilation) NOAA, ERA5, etc. 
Such data varies by geographic location, spatially, temporally, and spectrally. For handling and visualizing we chose to use [xarray](https://docs.xarray.dev/en/stable/), [pandas](https://pandas.pydata.org/), and [seaborn](https://seaborn.pydata.org/). 
[SQLite](https://www.sqlite.org/index.html) is used for information extraction from databases, [ARLreader](https://github.com/martin-rdz/ARLreader) is used to read the NOAA ARLs data.
Additional science codes are used for physics or machine learning models, as [SciPy](https://scipy.org/), [lidar_molecular](https://gitlab.com/ioannis_binietoglou/lidar_molecular) and more.
The learning section relies on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://www.pytorchlightning.ai/) and [RAY](https://www.ray.io/).
These are wonderful learning packages, if you are not familiar they have many tutorials. 

We are grateful to the developers and creators of the above libraries.
# Installation

To get the code simply clone it - 

`git clone https://github.com/Addalin/learning_lidar.git`

Then, to setup the environment - 
- `cd learning_lidar`
- `conda env create -f environment.yml`

Activate it by -
`conda activate lidar`

Run `python setup.py develop` to locally install the lidar learning package - 
this is not currently necessary but can assist with missing paths when running scripts from command line. 

## Running the scripts

Each script can be run separately. They all use the command line format, with the base arguments of 
`--station_name, --start_date, --end_date, --plot_results, --save_ds`, and additional agruments based on the specific script.

For example to run generation main:
`python generation_main.py --station_name haifa --start_date 2017-09-01 --end_date 2017-10-31 --plot_results --save_ds`

Where relevant, use the `--use_km_unit` flag to use km units vs m units.

## Code Structure

Under `learning_lidar`:

- [Preprocessing](#preprocessing)

- [Generation](#generation)

- [Dataseting](#dataseting)

- [Learning_phase](#learning_phase)

In general, each sub folder corresponds to a process, and each standalone script is in a different file, and has a corresponding `<script_name>_utils`
file for subroutines,

There is a general `utils` folder, and additional minor scripts and notebooks not mentioned here.




### Preprocessing
- Main script is `preprocessing/preprocessing.py`
- converts raw data into clean format. 
- Specifically can be used to:
  - download and convert gdas files with the `--download_gdas` and `--convert_gdas` flags
  - generate molecular `--generate_molecular_ds`, lidar `--generate_lidar_ds` or raw lidar `--generate_raw_lidar_ds`
  - `--unzip_lidar_tropos` to automatically unzip downloaded TROPOS lidar data.

### Generation

- Generates ALiDAn data. `generation/generation_main.py` is a wrapper for the different parts of the process and
and can be used to to run everything at once for a given period. It includes:
  - Background Signal (`genage_bg_signals`)
  - Angstrom Exponent and optical depth (`read_AERONET_data`)
  - KDE Estimation (`KDE_estimation_sample`)
  - Lidar Constant (`generate_LC_pattern`)
  - Density generation (`generate_density`)
  - signal generation (`daily_signals_generation`)

- Additional code:
  - Figures output and validation of ALiDAn [1] are under [generation/ALiDAn Notebooks](generation/ALiDAn Notebooks).
  - Large parts of the code were initially written as notebooks, then manually converted to py files. 
    - For example under `generation/legacy` are the original notebooks.
    - `generate_bg_signals` has been converted to py, 
     but not yet generalized to any time period, thus the original notebook is still in the main generation folder.
    - `overlap.ipynb` hasn't been converted to py yet. Overlap is an additional part of the generation process
    - Figures that were necessary for the paper are saved under the `figures` subdirectory. 
    Only relevant if the `--plot_results` flag is present.
  

### Dataseting
- Main script is `dataseting/dataseting.py`
- Flags:
  - Used to create a csv of the records - `--do_dataset`
  - `--extend_dataset` to add additional info to the dataset
  - Create calibration dataset from the extended df - `--do_calibration_dataset`
  - `--create_train_test_splits` to create train test splits
  - `--calc_stats` to calculate mean, min, max, std statistics
  - `--create_time_split_samples` to split up the dataset into small intervals.
  - Note, use `--generated_mode` to apply the operations on the generated data (vs the raw tropos data)
  
### Learning_phase
The learning pipeline is designed to receive two data types: raw lidar measurements by pollyXT and simulated by ALiDAn. 
The implementation is oriented to lidar calibration. However, one can easily apply any other model.

- Deep learning module to predict 'Y' given 'X'.
- Makes use of parameters from  [run_params.py](learning_lidar/learning_phase/run_params.py). 
- Configure the params as desired, then run the NN with `python main_lightning.py` 
- The models are implemented with PyTorch Lightning, currently only [calibCNN.py](learning_lidar/learning_phase/models/calibCNN.py).
- `analysis_LCNet_results` extracts the raw the results from a results folder and displays many comparisons of the different trials.
NOTE: currently `analysis_LCNet_results.ipynb` is old results with messy code. Updated code is at [analysis_LCNet_results_no_overlap.ipynb](learning_lidar/learning_phase/analysis_LCNet_results_no_overlap.ipynb)
and this is the notebook that should be used!
- [model_validation.py](learning_lidar/learning_phase/model_validation.py) is a script that was barely used yet but is meant to be used to load a pretrained model and 
use it to reproduce results.


## Notes 
1. The [data](data) folder contains both data necessary for the generation, and csv files that are created in the dataseting stage,
and needed as input for learning phase. Specifically -
   1. `stations.csv` defines stations, currently also relevant when working on a different computer.
   2. `dataset_<station_name>_<start_date>_<end_date>.csv` contain links to the actual data paths. Each row is a record.
2. There are many todos in the code, some of which are crucial for certain stages, and some 'nice to have'.
3. The [run_script.sh](learning_lidar/run_script.sh) can be used as an example of how to run parts of the code from the terminal with the commandline arguments,
for example for different dates.
4. [Paths_lidar_learning.pptx](assets/Paths_lidar_learning.pptx) is for the planned changes to the data paths - which
are meant to be much more organized, easier to maintain and less dependent.
5. The [pyALiDAn_dev](pyALiDAn_dev) - is a private folder of ongoing research. 
