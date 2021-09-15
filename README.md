# Installation

To get the code simply clone it - 

`git clone https://github.com/Addalin/learning_lidar.git`

Then, to setup the environment - 
- `cd learning_lidar`
- `conda env create -f environment.yml`

Activate it by -
`conda activate lidar`


## Running the scripts

Each script can be run separately. They all use the command line format, with the base arguments of 
`station_name, start_date, end_date, plot_results, save_ds`, and additional agruments based on the specific script.

For example to run generation main:
`python generation_main.py --station_name haifa --start_date 2017-09-01 --end_date 2017-10-31 --plot_results --save_ds`

Where relevant, use the `use_km_unit` flag to use km units vs m units.

## Code Structure

Under `learning_lidar`:

- [generation](generation)

- [dataseting](dataseting)

- [preprocessing](preprocessing)

- [learning_phase](learning_phase)

In general, each standalone script is in a different file, and has a corresponding `<script_name>_utils`
file for subroutines.

### generation

- Generates the different properties. `generation_main.py` is a wrappper for the different parts of the process and
and can be used to to run everything at once for a given period. It includes:
  - Background Signal
  - Angstrom Exponent and optical depth
  - KDE Estimation
  - Lidar Constant
  - Density generation
  - signal generation


### dataseting

- Used to create a csv of the records - `do_dataset`
- `extend_dataset` to add additional info to the dataset
- Create calibration dataset from the extended df - `do_calibration_dataset`
- `create_train_test_splits` to create train test splits
- `calc_stats` to calculate mean, min, max, std statistics
- `create_time_split_samples` to split up the dataset into small intervals.
- Note, use `generated_mode` to apply the operations on the generated data (vs the raw tropos data)


### preprocessing

- converts raw data into clean format. 
- Specifically can be used to:
  - download and convert gdas files with the `download_gdas` and `convert_gdas` flags
  - generate molecular `generate_molecular_ds`, lidar `generate_lidar_ds` or raw lidar `generate_raw_lidar_ds`
  - `unzip_lidar_tropos` to automatically unzip downloaded tropos lidar data.

### learning_phase

- deep learning module to predict y
- Makes use of parameters from `run_params.py`. 
- Configure the params as desired, then run the network with `python main_lightning.py` 


