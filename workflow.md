# Learning lidar preprocess

# 1. Preprocessing Workflow - for TROPOS measurements

## Preprocessing Tasks
0. Download gdas
1. Convert all gdas files in the directory `H:\data_haifa\DATA FROM TROPOS\GDAS\haifa` to txt files (using gdas2radiosonde()) - ***DONE*** 
2. Generate a daily molecular profile (2D) from the converted txt files  - ***DONE***
3. create a class of station(<location_name>) loaded from stations_info.csv - ***DONE*** 
4. Generate molecular nc files per given date , based on GDAS files  - ***DONE*** 
5. Generate Lidar nc files per given date, based on TROPOS files - ***DONE*** 

## TODOS and Open issues to figure out:
1. Troubleshooting of file and system errors (see TODOS in preprocessing.py) - ***DONE***
2. Fix erroneous gdas files (using ARLreder module) - ***DONE*** 
3. Debug what causes prep.generate_daily_molecular() to run so slowly:
    - Solution: run this process in parallel ***DONE*** 
    - Sending variables (flags) to prep.gen_daily_ds ( ) - ***OPEN***

## CLEANUP tasks:
Separate/merge and organize the following modules:
   1. preprocessing: 
      - cleanup- ***DONE***
      - TODOS - ***OPEN***
      - Explore dynamic range of range corrected signal - ***DONE***

   2. mic_lidar:
      - cleanup- ***DONE***
      - cleanup duplicated code - make sure only micLidar.py are used , and move `generate_atmosphere.py` to  `legacy code` ***DONE*** 
      - TODOS - not urgent - ***OPEN***

# 2. Dataseting Workflow

## Dataseting Tasks:  
1. Create a time slots list of samples from the database:
    [date, period, start_time, end_time, calib_method, lc, profile_file_name]  ***DONE*** 
2. For each row in the list create sample:   ***DONE*** 
    - use date, start_time, end_time to extract p_r2 and all other properties from `*att_bsc.nc`
    - use date, start_time, end_time to extract p_mol_r2 - from molecular profile
    - from profile_file_name extract: ro,r1
3. Create a list of time slots from the database file - ***DONE***
4. Create and save database samples - ***DONE*** 
5. Generate dataset of lidar and molecular to certain period - see `dataseting.py` ***DONE***
6. update create_dataset() to work for a list of days and check it on the generated dataset (above) ***DONE***
7. Split the dataset to have "test" and "train/valid" examples.  such test examples will be used only when needed. - ***DONE***
8. Generate a similar train/test dataset - from the generation procedure - ***DONE***
9. Calculate stats of the dataset :
    - Generating statistics for generated datasets ***DONE***
    - Generating statistics for TROPOS datasets ***OPEN***
10. Extended calibration dataset:
   - Incorporate beta max info per signal retrieval per a wavelength, from: `KDE_estimation_sample.ipynb` to: `dataseting.py` - in `extend_calibration_info()` - ***OPEN***
   - Update the `data_vars` in `create_calibration_ds()` similar to other parameters - ***OPEN***

## Database structure
Each sample contains the followings: 

#### 1. Properties: 
    
    - date          <datetime.date>
    
    - period        <timedelta.min>  - in minutes, e.g.: 30 min or 60 min 
    
    - start_time    <timedelta.time>
    
    - end_time      <timedelta.time>
    
    - wavelength    <float>          - in meters, e.g.: LAMBDA.UV, LAMBDA.G, LAMBDA.IR 
    
    - latitude      <float>          - in meters            
    
    - longitude     <float>          - in meters
    
    - altitude      <float>          - in meters
    
    - calib_method  <string>         - e.g.,: Klett_Method, AOD_Constrained_Method, Raman_Method 

    - example type  <string>         - Type of the example - train/valid or test

#### 2. X (measurements):
    - p_r2          <torch.float64>  - 2D, raw lidar measurment 

    - p_mol_r2      <torch.float64>  - 2D, calculated lidar molecular measurment 

#### 3. Y (estimated values):
    - lc            <torch.float64>  - scalar, lidar constant

    - r0            <torch.float64>  - scalar, min height of reference range

    - r1            <torch.float64>  - scalar, max height of reference range

    - bin_r0        < int >          - index of min reference height bin (r0)

    - bin_r1        < int >          - index of max reference height bin (r0)

# 3. Generation - workflow

## Current Workflow:
1. "Ingredients" generation:
   
    1. **Daily mean background signal**
       - Notebook: `generate_bg_signals.ipynb`.
       - The 4th part of notebook creates mean bg for every day in a given period.
       - There are some minor open TODOS to complete.
       - Output: nc file of generate mean bg is saved per month
            - Folder : `D:\data_haifa\GENERATION\bg_dataset`.
            - NC file name: `generated_bg_haifa_2017-01-01_2017-01-31.nc`
       - TODOS: 
            - clean and update imports ***DONE***
            - create module `generate_bg_signals.py` ***OPEN***
    2. **Daily Angstrom Exponent and Optical Depth**
       - Notebook: `read_AERONET_data.ipynb`
       - Read Aeronet measurement for a month, and generate per month, the angstrom and aerosols optical depth (aod) datasets.
       - Input: `D:\data_haifa\AERONET\20170901_20170930_Technion_Haifa_IL\20170901_20170930_Technion_Haifa_IL.lev20`
       - Outputs:  `D:\data_haifa\AERONET\20170901_20170930_haifa_ang.nc`, `D:\data_haifa\AERONET\20170901_20170930_haifa_aod.nc`
       - TODOS:
         - Adapt to run more month (at least October). Requires the download from AERONET site ( Look in `cameraNetwork` to see how this is done)
         - Create a python module - ***OPEN***
         - Run on two more months ***OPEN***
        
    3. **Initial parameters for density generation**
       - Notebook: `KDE_estimation_sample.ipynb`
       - Parameters: $r_ref$, $\beta_{532}^{max}$, $Angstrom_{355,532}$, $Angstrom_{532,1064}$,LR 
       - Input datasets: 
            - df_extended (created in `preprocessing.py`). File name: `.\dataset_haifa_2017-09-01_2017-10-31_extended.csv` (for ref height)
            - ds_profile - profile retrieval of TROPOS (for max beta) 
            - ds_ang - Angstrom dataset created in `read_AERONET_data.ipynb`. File name:`D:\data_haifa\AERONET\20170901_20170930_haifa_ang.nc`
            - df_A_LR -  Angstrom-Lidar dataset from pre-measured figure. File name: `C:\Users\addalin\Dropbox\Lidar\code\Angstrom_LidarRatio\plot_data\plot_data\data_points_with_error_bars.csv` 
       - Output: nc file containing the parameters above 
            - Folder:  `D:\data_haifa\GENERATION`
            - NC File name: `generated_density_params_haifa_2017-09-01_2017-09-30.nc`
       - TODOS: 
            - Need some cleanup and organising ***DONE***
            - Generate on more months (at least for Oct 2017), currently available only for september.
            - Create a python module that runs the parameters' creation (includes  `KDE_estimation_sample.ipynb` & `read_AERONET_data.ipynb`) ***OPEN***
   4. **Lidar Constant for a period**
      - Notebook: `generate_LC_pattern.ipynb`   
      - Creates varying Lidar Power for a period. Currently, done for sep-oct 2017. Values are manually initialized based on values found in `ds_extended` (created in `dataseting.py`).
      - Output:  `D:\data_haifa\GENERATION\generated_LC_haifa_2017-09-01_2017-10-31.nc`
      - TODOs: 
        - Generate data for a full year, use similar power pattern, but with some varying parameters as days (sample uniform day : 70-90 days), and power ( ssmple from a segment of values per wavelength) ***OPEN***
        - Split output nc files , per month  ***OPEN***
       
   5. **Density Generation:**
      - Notebook: `generate_density.ipynb`  
      - Generate daily backscatter and extinction of aerosols
      - Inputs: ds_month_params `generated_density_params_haifa_2017-09-01_2017-09-30.nc` (created in `KDE_estimation_sample.ipynb`)
      - Output: ds_aer per day as: `D:\data_haifa\GENERATION\aerosol_dataset\09\2017_09_02_Haifa_aerosol_check.nc`
      - **TODOS**: 
        - Create a python module - ***DONE***
        - Adapt the code to run automatically for a required period:
          * For sep-oct 2017 ***DONE*** 
          * For other times ***OPEN** - This is dependent on finishing the above stages.  
        - Signals' variability enrichment (TBD - These are writen in my real notebook:) - not urgent as the above. This is a topic to handle after first run of the CNN with new signals. ***OPEN***
        - Open TODOS in: `generate_density_utils.py`,`generate_density.py` - ***OPEN*** 
2. **Lidar Signal generation**:
- Notebook: `daily_signals_generation.ipynb`
- Generating lidar signals
- Workflow: 
    1. Calculates total backscatter and total extinction profile (of molecular and aerosols)
    2. Calculate averaged daily lidar power (p +  p_bg)
    3. Calculate Poisson measurement of the signal
    
- Inputs:
  1. ds_aer - as `D:\data_haifa\GENERATION\aerosol_dataset\09\2017_09_02_Haifa_aerosol.nc` (created in `generate_density.ipynb`)
  2. ds_mol - as `D:\data_haifa\DATA FROM TROPOS\molecular_dataset\2017\09\2017_09_18_Haifa_molecular.nc` (created in `preprocessing.py`)
  3. ds_bg  - as `D:\data_haifa\GENERATION\bg_dataset\generated_bg_haifa_2017-01-01_2017-01-31.nc` (created in `generate_bg_signals.ipynb`)
  4. ds_gen_p - `D:\data_haifa\GENERATION\generated_LC_haifa_2017-09-01_2017-10-31.nc` (created in `generate_LC_pattern.ipynb`)
- Output:  TBD
  
- **TODOS**:
  1. Massive cleanup of the file & check up that it works with the most recent version of the generated  density files - ***DONE***
  2. Create mean_pbg as 2D mean background power (This is going to be one of inputs X) ***DONE***
  3. Save NC of including the range corrected signal X (pr^2, mean_pbg), Y (LC) and p.  ***DONE***
  4. Adapt for running automatically on each day in a given period:
    * For sep-oct 2017 ***DONE*** 
    * For other times ***OPEN** - This is dependent on finishing the above stages of density creation. 
  5. Create a python module - ***DONE***
  6. Save range_corre after poisson procedure (without bg)  ***OPEN***
    
## Issues:
   1. Debug propagation of PLOT_RESULTS flag in `generate_density_utils.py` & `daily_signals_generations_utils.py`
   2. loggers :
      - Make sure the logger type can be passed as argument to main() - e.g. in `generate_density_utils.py` & `daily_signals_generations_utils.py`
      - Set loggers folder to be withing the submodule that created it.  


# 4. Learning system - workflow
##  Dataloader Tasks:
1. Adjust the train/valid and test loaders according to the split dataset version - ***DONE***
2. For Y loader - Convert height bins into heatmap - ***OPEN***
3. For X loader - convert xarray.dataset to pytorch and split into time slices - ***DONE***
## CNN Task:
1. Dataloder - ***DONE***
2. Create dataloader for generated data - ***DONE***
3. Add "accuracy" /relative loss of the result (AKA MARE loss) - ***DONE***
4. Updates parameters for tensorboars (TBD) 
    - split errors by the wavelength ***OPEN***
5. Filter the dataset - to train on a specific populations - ***DONE*** 
    - currently, works for features written in the dataset file
6. Use the calculated statistics of the database (mean , std) for Normalization - ***DONE***
7. LC Net Architecture - ***ON GOING***
   
    a. runs of hiden_sizes,fc_size
    
    * hiden_sizes : [2,2,2,2] , [3,3,3,3] , fc_size: [4,32,16] ***DONE***
    * hiden_sizes : [1,1,1,1] , [4,4,4,4] , fc_size: [4,32,16] ***DONE*** 
    * With Normaliztion - hiden_sizes : [1,1,1,1] ,[2,2,2,2], [3,3,3,3], [4,4,4,4] fc_size: [4,32,16], lr: 1e-3 ***OPEN***
    * poisson signal With Normaliztion - hiden_sizes : [1,1,1,1] ,[2,2,2,2], [3,3,3,3], [4,4,4,4] fc_size: [4,32,16], lr: 1e-3 ***ON Going***
    
    b. runs for testing inputs:
    * range corrected of `signal` and `lidar` - ***ON GOING*** 
    * range_corrected of `signal` after poisson noise ***OPEN***
    * Adding bg ***OPEN***
    
    c. Run LC net data on 2 more months (Jun & April 2017) - this requires creation of signals before. ***OPEN***
    
    d. Create summery of the runs in csv for each run include also : 
        number of parameters, run time.
        This summery should help up decide on LC net: 
        hiden sizes , fc size , type of input signal (what range_corrected should be use and if using bg) 

8. Load checkpoint - for more train epochs ***OPEN***
9. Run checkpoint on test dataset ***OPEN***
10. change results folder (D is better)  ***OPEN***
11. Resolve Errors & warnings during training ***OPEN***: 
    - [issue1](https://github.com/Addalin/learning_lidar/issues/28#issue-881010672) 
    - [issue2](https://github.com/Addalin/learning_lidar/issues/27#issue-881005719)
12. Keep only relevant files for training ( create a script that runs on the csv database that will pass only the relevant files) on the SSD disk. ***OPEN***

13. Decide if using auto encoder de-noiser  ***TBD***
14. Decide on RNN architecture ***TBD***

# Others
## Other repo and coding issues:

1. Data folder containing: 
    - TODO ADD TO GIT? - particals distribution (LR- A) : `C:\Users\addalin\Dropbox\Lidar\code\Angstrom_LidarRatio\plot_data\plot_data\data_points_with_error_bars.csv`. usage:`KDE_estimation_sample.ipynb`
    - TODO ADD TO GIT? - iradiance-angle  graph :  `C:\Users\addalin\Dropbox\Lidar\code\background_signal\irradiance_solarElevation.csv` , usage: `generate_bg_signals.ipynb`
    - bg mean curve : `C:\Users\addalin\Dropbox\Lidar\code\background_signal\curve_params.yml` usage: `generate_bg_signals.ipynb`
    - stations csv: `C:\Users\addalin\Dropbox\Lidar\code\stations.csv`
    - failed_gdas_files_path : `gdas2radiosonde_failed_files.csv`, usage `datasetting.py` & `fix_gdas_errors.py`
    - dataset for learinng dataloader: `.\dataset_haifa_2017-09-01_2017-10-31_on_D.csv` or `dataset_haifa_2017-09-01_2017-10-31.csv` - created in `datasetting.py`, usage in: `dataloader.py`
    - extended dataset used for analysis and generation: `dataset_haifa_2017-09-01_2017-10-31_extended.csv` & `dataset_haifa_2017-09-01_2017-10-31_extended.nc` - created in `datasetting.py`, usage in:`KDE_estimation_sample.ipynb`, `generate_profiles_from_measurments.ipynb`, `generate_profiles_from_TROPOS_retrievals.ipynb`
    - take care of relative paths for loading/saving the above files . ***OPEN***
2. Logging folder:
   - create folder per phase of the system
3. Utils folder: 
    * Currently contains:
      1. `bezier.py`
      2. `img_interp.py`
      3. `miscLidar.py`
      4. `smooth.py`
      5. `utils.py`
    * TODOS: Split utils folder to : 
      1. `misc_lidar.py` - lidar formulas (prev: miscLidar) ***DONE***
      2. `proc_utils.py` - processing and math utils (bezier, smooth, normalize & make_interpolated_image - from generate_density_utils)***DONE***
      3. `utils.py`- General system and run utils (what is currently utils.py)  ***DONE***
      4. `global_settings.py` ***ALMOST DONE*** required some clearness to vis_utils.py) 
      5. `vis_utils.py` - All visualizations styling and plot figures of xarray in 2D or 1D or 3D (surf) ***OPEN***
4. Dataset Folder:
   - data_station
        - AERONET (Requires rearrangement)
        - DATA FROM TROPOS : 
          - lidar_dataset (X)
          - molecular_dataset (X)
          - GDAS:
            - haifa (files form NOAA)
            - haifa_preproc (created in preproccesing.py) 
          - data ( FROM TROPOS)
        - GENERATION:
          - aerosol_dataset
          - bg_dataset
4. Update environment installations documents. as: 
   - ARLreader ***OPEN***
   - molecular/ lidar_molecular   - ask Ioannis Binietoglou, Mike Kottas - for link to updated repo inorder to cite their work. ***OPEN***
   - pytorch. ***OPEN***
5. Add version number to dataset files (with relation to git version). - currently will be solved as folder versions  ***OPEN***
6. merge visualization and plotting settings

## Possible issues that may require solutions: 
1. Save space for the saved data: split each dataset of range corrected to 2 : one contains the exponent of 10 , and the other the base. ***IGNORE***
2. Figure out the plot_range values of TROPOS dataset and how they were decided. ***IGNORE***

## Writing a description of the procedure for creating the signals - ***OPEN***
