# Installation

To get the code simply clone it - 

`git clone https://github.com/Addalin/learning_lidar.git`

Then, to setup the environment - 
- `cd learning_lidar`
- `conda env create -f environment.yml`

Activate it by -
`conda activate lidar`


Lidar molecular setup (place in project root directory):

    cd lidar_molecular    
    python setup.py install

# Project Structure

- [Source Code](code-structure)
- [Data]()



## Code Structure

Under `learning_lidar`:

- [generation](generation)

- [dataseting](dataseting)

- [preprocessing](preprocessing)

- [learning_phase](learning_phase)



### generation

- Generates data


### dataseting

- ?


### preprocessing

- converts raw data into clean format

### learning_phase

- deep learning module to predict y