# Learning lidar preprocess

# 1. Preprocessing Workflow - for TROPOS measurements

## Preprocessing Tasks
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
   2. micLidar:
      - cleanup- ***DONE***
      - cleanup duplicated code - make sure only micLidar.py are used , and move `generate_atmosphere.py` to  `legacy code` ***OPEN*** 
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
7. Split the dataset to have "test" and "train/valid" examples.  such test examples will be used only when needed. - ***OPEN***
8. Generate a similar train/test dataset - from the generation procedure - ***OPEN***
9. Incorporate beta max info per signal from `KDE_estimation_sample.ipynb` to `dataseting.py`
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

## CLEANUP tasks:
-

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
    
    2. **Daily Angstrom Exponent and Optical Depth**
       - Notebook: `read_AERONET_data.ipynb`
       - Read Aeronet measurement for a month, and generate per month, the angstrom and aerosols optical depth (aod) datasets.
       - Input: `D:\data_haifa\AERONET\20170901_20170930_Technion_Haifa_IL\20170901_20170930_Technion_Haifa_IL.lev20`
       - Outputs:  `D:\data_haifa\AERONET\20170901_20170930_haifa_ang.nc`, `D:\data_haifa\AERONET\20170901_20170930_haifa_aod.nc`
       - TODOS:
         - Adapt to run more month (at least October). Requires the download from AERONET site ( Look in `cameraNetwork` to see how this is done)

    3. **Initial parameters for density generation**
       - Notebook: `KDE_estimation_sample.ipynb`
       - Parameters: $r_ref$, $\beta_{532}^{max}$, $Angstrom_{355,532}$, $Angstrom_{532,1064}$,LR 
       - Input datasets: 
            - df_extended (created in `preprocessing.py`). File name: `.\dataset_haifa_2017-09-01_2017-10-31_extended.csv` (for ref height)
            - ds_profile - profile retrival od TROPOS (for max beta) 
            - ds_ang - Angstrom dataset created in `read_AERONET_data.ipynb`. File name:`D:\data_haifa\AERONET\20170901_20170930_haifa_ang.nc`
            - df_A_LR -  Angstrom-Lidar dataset from pre-measured figure. File name: `C:\Users\addalin\Dropbox\Lidar\code\Angstrom_LidarRatio\plot_data\plot_data\data_points_with_error_bars.csv` 
       - Output: nc file containing the parameters above 
            - Folder:  `D:\data_haifa\GENERATION`
            - NC File name: `generated_density_params_haifa_2017-09-01_2017-09-30.nc`
       - TODOS: 
            - Need some cleanup and organising
            - Generate on more months (at least for Oct 2017), currently available only for september.
   4. **Lidar Constant for a period**
      - Notebook: `generate_LC_pattern.ipynb`   
      - Creates varying Lidar Power for a period. Currently, done for sep-oct 2017. Values are manually initialized based on values found in `ds_extended` (created in `dataseting.py`).
      - Output:  `D:\data_haifa\GENERATION\generated_LC_haifa_2017-09-01_2017-10-31.nc`
      - TODOs: 
        - Generate data for a full year, use similar power pattern, but with some varying parameters as days (sample uniform day : 70-90 days), and power ( ssmple from a segment of values per wavelength)
        - Separate output files , per month
       
   5. **Density Generation:**
      - Notebook: `generate_density.ipynb`  
      - Generate daily backscatter and extinction of aerosols
      - Inputs: ds_month_params `generated_density_params_haifa_2017-09-01_2017-09-30.nc` (created in `KDE_estimation_sample.ipynb`)
      - Output: ds_aer per day as: `D:\data_haifa\GENERATION\aerosol_dataset\09\2017_09_02_Haifa_aerosol_check.nc`
      - **TODOS**: 
        - Adapt the code to run automatically for a required period. (at least sep-oct 2017)
        - Signals' variability enrichment (TBD - These are writen in my real notebook:) - not urgent as the above. This is a topic to handle after first run of the CNN with new signals.
2. **Lidar Signal generation**:
- Notebook: `daily_signals_generation.ipynb`
- Generating lidar signals
- Workfolow: 
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
  1. Massive cleanup of the file & check up that it works with the most recent version of the generated  density files
  2. Create mean_pbg as 2D mean background power (This is going to be one of inputs X)
  3. Save NC of including the range corrected signal X (pr^2, mean_pbg), Y (LC) and p. 
  4. Adapt for running automatically on each day in a given period (at least sep-oct 2017)
    
- Open issues: 
- Why the signal has appearance of stairs? Check also TROPOS measurements

## Tasks:
   1. 
   2. 
   3. 

# 4. Learning system - workflow
##  Dataloader Tasks:
1. Adjust the train/valid and test loaders according to the split dataset version - ***OPEN***
2. For Y loader - Convert height bins into heatmap - ***OPEN***
3. For X loader - convert xarray.dataset to pytorch and split into time slices - ***DONE***
## CNN Task:
1. Dataloder - ***DONE***
2. Create dataloader for generated data - ***OPEN***
3. Add "accuracy" of the result (AKA MARE loss) - ***OPEN***
4. Updates parameters for tensorboars (TBD)



4. Explore dynamic range of range corrected signal - done , waiting for TROPOS response


# Others
## Other repo and coding issues:
1. Data folder containing: 
    - particals distribution (LR- A) : `C:\Users\addalin\Dropbox\Lidar\code\Angstrom_LidarRatio\plot_data\plot_data\data_points_with_error_bars.csv`. usage:`KDE_estimation_sample.ipynb`
    - iradiance-angle  graph :  `C:\Users\addalin\Dropbox\Lidar\code\background_signal\irradiance_solarElevation.csv` , usage: `generate_bg_signals.ipynb`
    - bg mean curve : `C:\Users\addalin\Dropbox\Lidar\code\background_signal\curve_params.yml` usage: `generate_bg_signals.ipynb`
    - stations csv: `C:\Users\addalin\Dropbox\Lidar\code\stations.csv`
    - failed_gdas_files_path : `gdas2radiosonde_failed_files.csv`, usage `datasetting.py` & `fix_gdas_errors.py`
    - dataset for learinng dataloader: `.\dataset_haifa_2017-09-01_2017-10-31_on_D.csv` or `dataset_haifa_2017-09-01_2017-10-31.csv` - created in `datasetting.py`, usage in: `dataloader.py`
    - extended dataset used for analysis and generation: `dataset_haifa_2017-09-01_2017-10-31_extended.csv` & `dataset_haifa_2017-09-01_2017-10-31_extended.nc` - created in `datasetting.py`, usage in:`KDE_estimation_sample.ipynb`, `generate_profiles_from_measurments.ipynb`, `generate_profiles_from_TROPOS_retrievals.ipynb`
2. Logging folder:
   - create folder per phase of the system
3. Utils folder: 
    - `bezier.py`
    - `img_interp.py`
    - `miscLidar.py`
    - `smooth.py`
    - `utils.py`
2. Update environment installations documents (as ARLreader and pytorch ) ***OPEN***
3. Add version number to dataset files (with relation to git version). - currently will be solved as folder versions  ***OPEN***

## Possible issues that may require solutions: 
1. Save space for the saved data: split each dataset of range corrected to 2 : one contains the exponent of 10 , and the other the base. ***IGNORE***
2. Figure out the plot_range values of TROPOS dataset and how they were decided. ***IGNORE***
