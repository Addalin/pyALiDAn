import warnings

warnings.filterwarnings ( "ignore" )
import pandas as pd
import os
from datetime import datetime , time
import numpy as np

import torch , torchvision
import torch.utils.data
import xarray as xr
import preprocessing as prep
import torch.nn as nn
import time
from utils import create_and_configer_logger
from torch.utils.tensorboard import SummaryWriter



class lidarDataSet ( torch.utils.data.Dataset ) :
    """TODO"""

    def __init__ ( self , csv_file , transform = None,
                   top_height = 15.0 , Y_features = [ 'LC' , 'r0' , 'r1' ]) :
        """
        Args:
            csv_file (string): Path to the csv file of the database.
            :param transform:
        """
        self.data = pd.read_csv ( csv_file )
        self.key = [ 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                     'start_time_period' , 'end_time_period' ]
        self.Y_features = Y_features # [ 'LC' , 'r0' , 'r1' ]  # , 'LC_std'
        self.X_features = [ 'lidar_path' , 'molecular_path' ]
        self.profiles = [ 'range_corr' , 'attbsc' ]
        self.top_height = top_height
        self.transform = transform

    def __len__ ( self ) :
        return len ( self.data )

    def __getitem__ ( self , idx ) :
        # load data
        X = self.load_X ( idx )
        Y = self.load_Y ( idx )

        sample = {'x' : X , 'y' : Y}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_splits( self, n_test = 0.2 ):
        test_size = round (n_test * len(self))
        train_size = len(self) - test_size
        train_set , test_set = torch.utils.data.random_split ( self , [ train_size , test_size ] )
        return train_set , test_set

    def load_X( self, idx ):
        """
        Returns X samples - measurements of lidar, and molecular
        :param idx: index of the sample
        :return: A list of two element each is of type xarray.core.dataarray.DataArray.
        0 - is for lidar measurements, 1 - is for molecular measurements
        """
        row = self.data.loc [ idx , : ]

        # Load X datasets
        X_paths = row [ self.X_features ]
        datasets = [ prep.load_dataset ( path ) for path in X_paths ]

        # Calc sample height and time slices
        hslices = [
            slice ( ds.Height.min ( ).values.tolist ( ) , ds.Height.min ( ).values.tolist ( ) + self.top_height )
            for ds in datasets ]
        tslice = slice ( row.start_time_period , row.end_time_period )

        # Crop slice from the datasets
        X_ds = [ ds.sel ( Time = tslice , Height = hslice ) [ profile ]
                 for ds , profile , hslice in zip ( datasets , self.profiles , hslices ) ]

        return X_ds

    def load_Y( self, idx ):
        """
        Returns Y features for estimation
        :param idx: index of the sample
        :return: pandas.core.series.Series
        """

        row = self.data.loc [ idx , : ]
        Y = row [ self.Y_features ]
        return Y


class PowTransform(object):
    def __init__( self,powers = {'range_corr' : 0.5 , 'attbsc' : 0.5 ,
                             'LC' : 0.5 , 'LC_std' : 0.5 , 'r0' : 1 , 'r1' : 1} ):
        self.Y_features = [ 'LC' , 'r0' , 'r1' ]  # , 'LC_std'
        self.profiles = [ 'range_corr' , 'attbsc' ]
        self.X_powers = [ powers [ profile ] for profile in self.profiles ]
        self.Y_powers = [ powers [ feature ] for feature in self.Y_features ]

    def __call__ ( self , sample ) :
        X , Y = sample [ 'x' ] , sample [ 'y' ]
        X = [ self.pow_X(x_i, pow_i) for (x_i, pow_i) in zip ( X, self.X_powers)]
        Y  = self.pow_Y(Y)
        return {'x' : X , 'y' : Y}

    def pow_X ( self , x_i, pow_i ) :
        """

        :param x_i: xr.dataset: a lidar or a molecular dataset
        :return: The dataset is raised (shrink in this case) by the powers set.
        Acts similarly to gamma correction aims to reduce the input values.
        """
        # trim negative values
        x_i = x_i.where ( x_i >= 0 , np.finfo ( np.float ).eps )
        # apply power - using apply_ufunc function to accelerate
        x_i = xr.apply_ufunc ( lambda x : x ** pow_i , x_i , keep_attrs = True )
        return x_i

    def pow_Y ( self , Y ) :
        """

        :param Y: pandas.core.series.Series of np.float values to be estimates (as LC, ro, r1)
        :return: The values raised by the relevant powers set.
        """
        return [ y_i ** pow for (pow , y_i) in zip ( self.Y_powers , Y ) ]


class ToTensor ( object ) :
    """Convert a lidar sample {x,y}  to Tensors."""

    def __call__ ( self , sample ) :
        X , Y = sample [ 'x' ] , sample [ 'y' ]


        # convert X from xr.dataset to concatenated a np.ndarray, and then to torch.tensor
        X = torch.dstack ( (torch.from_numpy ( X [ 0 ].values ) ,
                            torch.from_numpy ( X [ 1 ].values )) )
        # swap channel axis
        # numpy image: H x W x C
        # torch image: C X H X W
        X = X.permute ( 2 , 0 , 1 )

        # convert Y from pd.Series to np.array, and then to torch.tensor
        Y = torch.from_numpy ( np.array ( Y ).astype ( np.float32 ) )

        return {'x' : X ,'y' : Y}

# TODO : apply poisson noise (miscLidar.generate_poisson_signal(mu, n)), add gaussian noise for augmentation



class DefaultCNN(nn.Module):

    def __init__(self, in_channels=2, output_size=3, hidden_sizes=[16,32,64]):
        super(DefaultCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_sizes[0], kernel_size=(5,3), padding=3),
            nn.BatchNorm2d(hidden_sizes[0]),
            nn.ReLU(inplace = True),
            nn.Dropout2d ( p = 0.15 ),
            nn.MaxPool2d ( kernel_size = (4,2) , stride = (4,2) ),

            # Conv layer 2
            nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_sizes[1]),
            nn.ReLU(inplace = True),
            nn.Dropout2d ( p = 0.15 ) ,
            nn.MaxPool2d(kernel_size = (4,2), stride =(4,2)),

            # Conv layer 2
            nn.Conv2d ( in_channels = hidden_sizes [ 1 ] , out_channels = hidden_sizes [ 2 ] , kernel_size = 3 ,
                        padding = 1 ) ,
            nn.BatchNorm2d ( hidden_sizes [ 2 ] ) ,
            nn.ReLU ( inplace = True ) ,
            nn.Dropout2d ( p = 0.15 ) ,
            nn.MaxPool2d ( kernel_size = (4 , 2) , stride = (4 , 2) ) ,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear ( 8*32*8 , 512 ),
            nn.ReLU ( inplace = True ) ,
            nn.Dropout ( p = 0.1 ) ,
            nn.Linear ( 512 , output_size ) ,
        )

    def forward(self, x):
        #conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.shape[0], -1)

        # fc layer # TODO: add option to change in_channels to fc_layer (when changing hidden sizes of CNN)
        out = self.fc_layer(x)

        return out
def calculate_accuracy (model, loader,device):
    """

    :param model:
    :param loader:
    :param device:
    :return:
    """
    model.eval ( )             # Evaluation mode
    criterion = nn.MSELoss ( )
    running_loss = 0.0
    #with torch.no_grad: #TODO: why this is not working?
    for i,sample in enumerate(loader):
        x = sample [ 'x' ].to ( device )
        y = sample [ 'y' ].to ( device )
        y_pred = model(x)
        loss = criterion(y,y_pred)
        running_loss += loss.data.item ( )
    running_loss /= len ( loader )
    return  running_loss


def main( station_name = 'haifa' , start_date = datetime ( 2017 , 9 , 1 ) , end_date = datetime ( 2017 , 9 , 2 ),
          run_params={'model_n':0, 's_model':0,'powers':None,'lr':1e-3,'batch_size':4, 'Y_features':[ 'LC' , 'r0' , 'r1' ]} ):
    #logger = create_and_configer_logger ( 'data_loader.log' ) #TODO: Why this is causing the program to fall?
    # device - cpu or gpu?
    device = torch.device ( "cuda:0" if torch.cuda.is_available ( ) else "cpu" )
    print ( f"Using device: {device}")


    # Step 1. Load Dataset
    csv_path = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}_on_D.csv"
    powers = run_params['powers']
    # TODO: add option - y = {bin(r0),bin(r1)}
    lidar_transforms = torchvision.transforms.Compose([PowTransform(powers),ToTensor()]) if powers\
                  else torchvision.transforms.Compose([ToTensor()])
    dataset = lidarDataSet ( csv_path , lidar_transforms, top_height = 15.3, Y_features = run_params['Y_features'] )
    train_set , test_set = dataset.get_splits(n_test = 0.2)

    # Step 2. Create Model Class

    # Hyper parameters
    model_n = run_params['model']
    s_model = run_params['s_model']
    in_channels = 2
    output_size = len(run_params['Y_features'])
    hidden_sizes = [ 16, 32, 8] # TODO: add option - hidden_sizes = [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
    batch_size = run_params['batch_size']
    n_iters = 3000
    epochs = int(round(n_iters / (len ( train_set ) / batch_size)))
    learning_rate = run_params['lr']
    print (f"Start train model. Hidden size:{hidden_sizes}, batch_size:{batch_size}, n_iters:{n_iters}, epochs:{epochs}")
    use_pow = 'use_' if powers else 'no_'
    run_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_v{model_n}.{s_model}.{use_pow}pow_epochs_{epochs}_batch_size_{batch_size}_lr_{learning_rate}"


    model = DefaultCNN(in_channels = in_channels,output_size = output_size,hidden_sizes = hidden_sizes).to(device)
    numParams = 0
    for parameter in model.parameters ( ) :
        if (parameter.requires_grad) :
            numParams += parameter.numel ( )
    print( f"Number of parameters in model: {numParams}" )


    # Step 3. Instantiate Loss Class
    criterion = nn.MSELoss()

    # Step 4. Instantiate Optimizer Class
    optimizer = torch.optim.Adam ( model.parameters ( ) , lr = learning_rate )

    # Step 5. Initiate dataloader
    train_loader = torch.utils.data.DataLoader (
        train_set , batch_size = batch_size , shuffle = True , num_workers = 7 )

    test_loader = torch.utils.data.DataLoader (
        test_set , batch_size = batch_size , shuffle = False , num_workers = 7 )

    writer = SummaryWriter ( os.path.join('runs',run_name ))
    # Training loop
    for epoch in range(1, epochs+1):
        model.train()   # Training mode
        running_loss = 0.0
        epoch_time = time.time ( )
        for i_batch , sample_batched in enumerate ( train_loader ) :
            # get inputs of x,y & send to device
            x = sample_batched [ 'x' ].to ( device )
            y = sample_batched [ 'y' ].to ( device )

            # forward + backward + optimize
            y_pred = model(x)           # Forward
            loss = criterion(y,y_pred)  # set loss calculation
            optimizer.zero_grad()   # zeroing parameters gradient
            loss.backward()         # backpropagation
            optimizer.step()        # update parameters

            # for statistics
            global_iter = len(train_loader)*(epoch-1)+i_batch
            writer.add_scalar ( "batch_loss" , loss ,global_iter )
            running_loss += loss.data.item()

        # Normalizing loss by the total batches in train loader
        running_loss/=len(train_loader)
        writer.add_scalar ( "running_loss" , running_loss , epoch )

        # Calculate training and test set MSEloss for current model
        train_loss = calculate_accuracy(model, train_loader,device)
        writer.add_scalar ( "train_loss" , train_loss , epoch )
        test_loss = calculate_accuracy(model,test_loader,device)
        writer.add_scalar ( "test_loss" , test_loss , epoch )
        epoch_time = time.time ( ) - epoch_time
        log = f"Epoch: {epoch} | Running MSE Loss: {running_loss:.4f} |" \
              f" Training MSE loss: {train_loss:.3f} | Test MSE loss: {test_loss:.3f} | " \
              f" Epoch Time: {epoch_time:.2f} secs"
        # print()
        print( log )

        # Save the model

        state = {
            'model_state_dict' : model.state_dict ( ) ,
            'optimizer_state_dict' : optimizer.state_dict() ,
            'epoch': epoch,
            'batch_size':batch_size,
            'global_iter':global_iter,
            'n_iters':n_iters,
            'learning_rate': learning_rate
        }
        modelv_dir = f"checkpoints/model_{model_n}"
        if not os.path.isdir ( modelv_dir ) :
            os.mkdir (modelv_dir )
        model_fname = f"{datetime.now ( ).strftime ( '%Y_%m_%d_%H_%M' )}_" \
                      f"v{model_n}.{s_model}.{use_pow}pow_epoch_{epoch}-{epochs}_batch_size_{batch_size}_lr_{learning_rate}.pth"
        torch.save ( state , os.path.join(modelv_dir,model_fname) )
        print ( f"Finished training epoch {epoch}, saved model to: {model_fname}" )
    writer.flush ( )


if __name__ == '__main__' :
    station_name = 'haifa'
    start_date = datetime ( 2017 , 9 , 1 )
    end_date = datetime ( 2017 , 10 , 31 )
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [8]
    Y_features = [['r0' , 'r1'],['r0' , 'r1' , 'LC']]
    powers = [None,{'range_corr' : 0.5 , 'attbsc' : 0.5 , 'LC' : 0.5 , 'LC_std' : 0.5 , 'r0' : 1 , 'r1' : 1}]
    model_n = 0
    for s_model,yf in enumerate(Y_features):
        for lr in learning_rates:
            for pow in powers :
                for batch_size in batch_sizes:
                    run_params = {'model':model_n, 's_model':s_model ,'powers':pow,
                                  'Y_features': yf, 'lr':lr, 'batch_size':batch_size}
                    main ( station_name , start_date , end_date, run_params )