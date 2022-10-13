# Clearing pyALiDAn code
> Following a meeting with Shubi on 13/Oct/2022
> 
> This filee contains steps and information regarding clearing the code and publishing it in a public repository names pyALiDAn

## The chosen solution to publish a clear repo 
1. Create a folder that the original repo is cloned into it:

   `git clone https://github.com/Addalin/learning_lidar.git C:\Users\addalin\Dropbox\Lidar\pyALiDAn_public` 
   
   *Note: The creation of a new folder is aimed to avoid faults in file deletion that may not be recovered! (especially of uncommitted files)* 
3. Create a new project that is related to this folder and set the same interperter used in the original project. i.e., `lidar_local_new`
4. Create a new branch named `public` 
5. Add a new remote repo using the following steps: 

   1. In pycharm choose `Git` -> Go to `Manage Remotes..` -> and Push `+`
   4. Set a new name, e.g., `public` and past the URL of the public git repo, e.g. https://github.com/Addalin/pyALiDAn.git

6. Make all changes required.
7. Push changes to the `public` branch. 
8. Push changes to the new remote channel, using the following steps:
   1. In order to push to the remote channel Go to `Git` -> `Fetch`
   2. Then push to the remote channel:
      1. `Git` -> `Push`
      2. Change the channel name from `orign` to `public` 
      3. Change the branch name from `public` to `main`

### Warnings
**Make sure that the project that is open in pycharm is `pyALiDAn_public`!**

**Make sure you are working on the `public` branch!**

**Do not ever make a pull request from `public` to `master` in the original repo**

## List of candidate files to be deleted
### 1. The folder `Analysis` 
This folder is now aimed for RND and ongoing research.

Notebooks for publish from this folder were moved to a new folder named `learning_lidar/generation/ALiDAn Notebooks`.
### 2. Files in the main folder
1. `Paths_lidar_learning.pptx` 
2. `workflow.md`

### 3. Files in  `learning_lidar/dataseting/`
1. `update_signal_database.ipynb`

### 4. Files in  `learning_lidar/generation/`
1. The folder named `legacy`

### 5. Files in `learning_lidar/learning_phase`
1. `analysis_LCNet_results.ipynb`
2. `Manipulating_NN_inputs.ipynb`

### 6. Files in `learning_lidar/preprocessing`
1. `read_grib_files.ipynb`

## TODO:
1. Merge changes from servers to master and decide if they are relevant for publishing
2. Complete the list of files to be deleted
3. After pushing pyALiDAn create make is citable: https://guides.lib.berkeley.edu/citeyourcode