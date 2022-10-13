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

## List of files to delete
### Files in `pyALiDAn_public\Analysis`
1. Compare_PollyNet_results_on_ALiDAn_data.ipynb
2. 

## TODO:
1. complete the files to be deleted
2. after pushing pyALiDAn create make is citable: https://guides.lib.berkeley.edu/citeyourcode