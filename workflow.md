# Dataset preprocess

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

#### 2. X (measurements):
    - p_r2          <torch.float64>  - 2D, raw lidar measurment 

    - p_mol_r2      <torch.float64>  - 2D, calculated lidar molecular measurment 

#### 3. Y (estimated values):
    - lc            <torch.float64>  - scalar, lidar constant

    - r0            <torch.float64>  - scalar, min height of reference range

    - r1            <torch.float64>  - scalar, max height of reference range
    
## Workflow

1. Create a time slots list of samples from the database:
    [date, period, start_time, end_time, calib_method, lc, profile_file_name]
2. For each row in the list create sample: 
    - use date, start_time, end_time to extract p_r2 and all other properties from `*att_bsc.nc`
    - use date, start_time, end_time to extract p_mol_r2 - from molecular profile
    - from profile_file_name extract: ro,r1 
     
## Tasks
1. Convert all gdas files in the directory `H:\data_haifa\DATA FROM TROPOS\GDAS\haifa` to txt files (using gdas2radiosonde()) - done 
2. Generate a daily molecular profile (2D) from the converted txt files  - done
3. create a class of station(<location_name>) loaded from stations_info.csv - done 
4. Create a list of time slots from the database file - done 
5. Create and save database samples (dataloader) - done 
6. Generate dataset of lidar and molecular to certain period - on going 
7. update create_dataset() to work for a list of days and check it on the generated dataset (above)
8. For Y loader - Convert height bins into heatmap
9. For X loader - convert xarray.dataset to pytorch and split into time slices 

## TODOS and Open issues to figure out:
1. Troubleshooting of file and system errors (see TODOS in preprocessing.py) - done
2. Fix erroneous gdas files (using ARLreder module) - done 
3. Debug what causes prep.generate_daily_molecular() to run so slowly
4. Explore dynamic range of range corrected signal - done , waiting for TROPOS response

## General coding tasks:
1. Separate and organize the modules:  preprocessing, micLidar , dataloader 
2. Update environment installations documents (as ARLreader and pytorch )

## Possible issues that may require solutions: 
1. Save space for the saved data: split each sataset of range corrected to 2 : one contains the exponent of 10 , and the other the base (less importanat for now)
2. Figure out the plot_range values of TROPOS dataset and how they were decided
