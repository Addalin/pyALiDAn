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
1. Convert all gdas files in the directory `H:\data_haifa\DATA FROM TROPOS\GDAS\haifa` to txt files (using gdas2radiosonde())
2. Generate a daily molecular profile (2D) from the converted txt files  
3. create a class of station(<location_name>) loaded from stations_info.csv
4. Create a list of time slots from the database file
5. Create and save database samples
6. separate and organize the modules:  preprocessing, micLidar , dataloader 

## TODOS and Open issues to figure out:
1. Troubleshooting of file and system errors (see TODOS in preprocessing.py)
2. Fix erroneous gdas files (using ARLreder module)
