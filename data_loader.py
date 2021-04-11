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
from torch import functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss,_Reduction
from torch.overrides import has_torch_function , handle_torch_function
torch.manual_seed(8318)
import json

class lidarDataSet ( torch.utils.data.Dataset ) :
    """TODO"""

    def __init__ ( self , csv_path , transform = None,
                   top_height = 15.0 , Y_features = [ 'LC' , 'r0' , 'r1' ], wavelengths = [355,532,1064]) :
        """
        Args:
            csv_file (string): Path to the csv file of the database.
            :param transform:
        """
        self.data = pd.read_csv ( csv_path )
        self.key = ['idx', 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                     'start_time_period' , 'end_time_period' ]
        self.Y_features = Y_features
        self.X_features = [ 'lidar_path' , 'molecular_path' ]
        self.wavelengths = wavelengths  # TODO: make option to load data by desired wavelength/s
        self.profiles = [ 'range_corr' , 'attbsc' ]
        self.top_height = top_height
        self.transform = transform

    def __len__ ( self ) :
        return len ( self.data )

    def __getitem__ ( self , idx ) :
        # load data
        X = self.load_X ( idx )
        Y = self.load_Y ( idx )
        wavelength = self.get_key_val( idx, 'wavelength' ).astype(np.int32)

        sample = {'x' : X , 'y' : Y}
        if self.transform:
            sample = self.transform(sample)
            wavelength = torch.from_numpy(np.asarray(wavelength))
        sample['wavelength']= wavelength
        sample['idx'] = idx
        return sample

    def get_splits( self, n_test = 0.2, n_val = 0.2 ):
        test_size = round (n_test * len(self))
        val_size = round(n_val*len(self))
        train_size = len(self) - val_size - test_size
        train_set , val_set, test_set = torch.utils.data.random_split ( self , [ train_size ,val_size, test_size ] )
        return train_set ,val_set, test_set

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

    def get_key_val( self, idx, key='idx' ):
        """

        :param idx: index of the sample
        :param key: key is any of self.key values. e.g, 'idx', 'wavelength' etc...
        :return: The values of the required key for the sample
        """
        row = self.data.loc [ idx , : ]
        key_val = row[self.key.index(key)]
        return key_val

class PowTransform(object):
    def __init__( self,powers = {'range_corr' : 0.5 , 'attbsc' : 0.5 ,
                             'LC' : 0.5 , 'LC_std' : 0.5 , 'r0' : 1.0 , 'r1' : 1.0,'dr':1.0} ):
        # TODO: Pass Y_features to the constructor
        self.Y_features = [ 'LC' , 'r0' , 'r1' ,'dr']
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

class TrimNegative(object):
    def __init__( self):
        pass

    def __call__ ( self , sample ) :
        X , Y = sample [ 'x' ] , sample [ 'y' ]
        # trim negative values
        X  = [x_i.where ( x_i >= 0 , np.finfo ( np.float ).eps ) for x_i in X]
        return {'x' : X , 'y' : Y}


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
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        #conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.shape[0], -1)

        # fc layer # TODO: add option to change in_channels to fc_layer (when changing hidden sizes of CNN)
        out = self.fc_layer(x)

        return out


def calculate_statistics(model,criterion, run_params, loader,device, Y_features=None, wavelengths = None):
    """

    :param model:torch.nn.Module
    :param criterion: loss criterion function
    :param run_params: dict of running parameters
    :param loader: torch.utils.data.DataLoader
    :param device: torch.device ()
    :param Y_features: features list to calculate loss separately, for debug (this is an optional input)
    :param wavelengths: wavelengths list to calculate loss separately, for debug (this is an optional input)
    :return: stats - a dict containing train/validation loss criterion, and separately feature losses (for debug)
    """
    model.eval ( )        # Evaluation mode
    criterion = criterion

    if Y_features:
        # Initializing FeatureLoss - For debug. this loss is not affecting the model.
        count_less = {}
        feature_loss = {}
        for idx_f , feature in enumerate ( Y_features ) :
            feature_loss.update ( {feature : {}} )
            feature_loss [ feature ].update ( {'all' : 0.0} )
            for wav in wavelengths :
                feature_loss [ feature ].update ( {wav : 0.0} )
                count_less.update({wav:0.0})

    running_loss = 0.0
    with torch.no_grad():
        for i,sample in enumerate(loader):
            x = sample [ 'x' ].to ( device )
            y = sample [ 'y' ].to ( device )
            y_pred = model(x)
            loss = criterion(y,y_pred)
            running_loss += loss.data.item ( )

            if Y_features:
                wavelength = sample['wavelength']
                for idx_f , feature in enumerate ( Y_features ) :
                    feature_loss [ feature ][ 'all' ] += mare_loss ( y_pred[ : , idx_f ] , y[ : , idx_f ] ).data.item()
                    for wav in wavelengths :
                        idx_w = torch.where ( wavelength == wav )[0]
                        if idx_w.numel()>0 :
                           feature_loss [ feature ] [ wav ] += \
                                mare_loss (  y_pred [ idx_w ][ : , idx_f ] , y[idx_w][ : , idx_f ]).data.item()
                        else:
                            count_less[wav]+=1
    running_loss /= len ( loader )
    stats = {f"{run_params['loss_type']}": running_loss}
    if Y_features:
        for idx_f , feature in enumerate ( Y_features ) :
            feature_loss [ feature ] [ 'all' ] /= len ( loader )
            for wav in wavelengths:
                feature_loss [ feature ] [ wav ] /= (len( loader ) - count_less[wav ])
        stats.update( {'FeatureLoss':feature_loss} )
    return  stats


def mare_loss ( input , target , size_average = None , reduce = None , reduction = 'mean' ) :
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor

    r""" Overriding l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute relative value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    if not torch.jit.is_scripting ( ) :
        tens_ops = (input , target)
        if any ( [ type ( t ) is not Tensor for t in tens_ops ] ) and has_torch_function ( tens_ops ) :
            return handle_torch_function (
                mare_loss , tens_ops , input , target , size_average = size_average , reduce = reduce ,
                reduction = reduction )
    if not (target.size ( ) == input.size ( )) :
        warnings.warn ( "Using a target size ({}) that is different to the input size ({}). "
                        "This will likely lead to incorrect results due to broadcasting. "
                        "Please ensure they have the same size.".format ( target.size ( ) , input.size ( ) ) ,
                        stacklevel = 2 )
    if size_average is not None or reduce is not None :
        reduction = _Reduction.legacy_get_string ( size_average , reduce )

    expanded_input , expanded_target = torch.broadcast_tensors ( input , target )
    loss = torch.mean ( torch.abs ((expanded_target - expanded_input)/expanded_target) )
    #return torch._C._nn.l1_loss ( expanded_input/expanded_target , torch.div(expanded_target/expanded_target) , _Reduction.get_enum ( reduction ) )
    return  loss

class MARELoss(_Loss):
    r"""Creates a criterion that measures the mean absolute relative error (MARE) between each element in
    the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \frac{\left| x_n - y_n\right|}{\left| y_n \right|},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MARELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MARELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mare_loss(input, target, reduction=self.reduction)


def update_stats(writer, model,loaders ,device,run_params,criterion, epoch):
    """
    Update current epoch's state to writer
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param model: torch.nn.Module
    :param loaders: a list of [train_loader,val_loader], each of type torch.utils.data.DataLoader
    :param device:  torch.device ()
    :param run_params: dict of running parameters
    :param criterion: loss criterion function
    :param epoch: current epoch
    :return: curr_loss - current loss for train and for validation sets
    """
    curr_loss = {}
    feature_loss={}
    for loader, mode in zip(loaders,['train','val']):
        # calc current epoch statistics: train and val model, and feature loss for debug.
        stats = calculate_statistics ( model , criterion, run_params,loader , device,
                                       Y_features =run_params['Y_features'],
                                       wavelengths = run_params['wavelengths'] )

        # add loss for current epoch's model
        field_name = f"{run_params['loss_type']}/{mode}"
        field_value =stats [f"{run_params['loss_type']}"]
        writer.add_scalar ( field_name, field_value, epoch )
        curr_loss.update( {mode: field_value} )

        # add feature losses for debug
        feature_loss.update({mode:{}})
        for feature in run_params [ 'Y_features' ] :
            for wav in run_params [ 'wavelengths' ] :
                # add FeatureLoss (per wavelength per feature) , currently the metric is MAERLoss.
                # For debug , this loss is not affecting the model.
                field_name = f"FeatureLoss_{mode}/{feature}_{wav}"
                field_value = stats['FeatureLoss'] [ feature ] [ wav ]
                writer.add_scalar ( field_name , field_value , epoch )

            # add common FeatureLoss, currently the metric is MAERLoss.
            # For debug , this loss is not affecting the model.
            field_name = f"FeatureLoss_{mode}/{feature}"
            field_value = stats ['FeatureLoss'] [ feature] [ 'all' ]
            writer.add_scalar ( field_name , field_value , epoch )
            feature_loss[mode].update({feature:field_value})
    return curr_loss,feature_loss


def write_hparams(writer,run_params,run_name, cur_loss, best_loss, cur_loss_feature,best_loss_feature):

    results = {'hparam_last/loss_train' : cur_loss [ 'loss' ] [ 'train' ] ,
               'hparam_last/loss_val' : cur_loss [ 'loss' ] [ 'val' ],
               'hparam_last/epoch' : cur_loss [ 'epoch' ] ,
               'hparam_best/loss_train' : best_loss [ 'loss' ] [ 'train' ] ,
               'hparam_best/epoch_train' : best_loss [ 'epoch' ] [ 'train' ],
               'hparam_best/loss_val' : best_loss [ 'loss' ] [ 'val' ] ,
               'hparam_best/epoch_val' : best_loss [ 'epoch' ] [ 'val' ] ,
               }
    for mode in ['train','val']:
        for feature in run_params['Y_features']:
            results.update({f"{feature}_last/loss_{mode}" : cur_loss_feature [ 'loss' ][mode][ feature ]} )
            results.update({f"{feature}_last/epoch_{mode}" : cur_loss_feature [ 'epoch' ]} )
            results.update({f"{feature}_best/loss_{mode}":best_loss_feature['loss'][ mode ][feature]})
            results.update({f"{feature}_best/epoch_{mode}":best_loss_feature['epoch'][ mode ][feature]})

    run_params [ 'Y_features' ] = '_'.join ( run_params [ 'Y_features' ] )
    run_params [ 'hidden_sizes' ] = '_'.join ( [ str(val) for val in run_params [ 'hidden_sizes' ]] )
    run_params [ 'wavelengths' ] = '_'.join ( [ str(val) for val in run_params [ 'wavelengths' ]] )
    run_params ['powers'] = run_params['powers']['LC'] if run_params['powers'] is not None else 1

    writer.add_hparams(hparam_dict = run_params,metric_dict =results,run_name=run_name )

def get_model_dirs(run_name, model_n,s_model ):
    model_dir = f"model_{model_n}"
    submodel_dir = f"model_{model_n}.{s_model}"
    main_dir = os.path.join( os.getcwd() ,"cnn_models" , model_dir , submodel_dir ,run_name)
    checkpoints_dir = os.path.join(main_dir ,"checkpoints" )
    run_dir = os.path.join( main_dir ,"run" )
    if not os.path.isdir ( checkpoints_dir ) :
        os.makedirs ( checkpoints_dir )
    if not os.path.isdir ( run_dir ) :
        os.makedirs ( run_dir )
    return {'main': main_dir, 'checkpoints':checkpoints_dir, 'run':run_dir}

def main( station_name = 'haifa' , start_date = datetime ( 2017 , 9 , 1 ) , end_date = datetime ( 2017 , 9 , 2 ),
          run_params={'model_n':0, 's_model':0,'hidden_sizes':[ 16 , 32 , 8 ],
                      'powers':None,'loss_type' :'MSELoss','lr':1e-3,'batch_size':4,
                      'Y_features':[ 'LC' , 'r0' , 'r1'] , 'wavelengths': [355,532,1064]} ):
    #logger = create_and_configer_logger ( 'data_loader.log' ) #TODO: Why this is causing the program to fall?

    # device - cpu or gpu?
    device = torch.device ( "cuda:0" if torch.cuda.is_available ( ) else "cpu" )
    print ( f"Using device: {device}")


    # Step 1. Load Dataset
    csv_path = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}_on_D.csv"
    powers = run_params['powers']
    lidar_transforms = torchvision.transforms.Compose([PowTransform(powers),ToTensor()]) if powers\
                  else torchvision.transforms.Compose([ToTensor()])
    dataset = lidarDataSet ( csv_path , lidar_transforms , top_height = 15.3 ,
                             Y_features = run_params [ 'Y_features' ] )
    train_set , val_set, _ = dataset.get_splits(n_test = 0.2, n_val = 0.2)

    # Step 2. Create Model Class

    # Hyper parameters
    model_n = run_params['model']
    s_model = run_params['s_model']
    in_channels = 2
    output_size = len(run_params['Y_features'])
    hidden_sizes = run_params['hidden_sizes']
    batch_size = run_params['batch_size']
    n_iters = run_params['n_iters']
    loss_type = run_params['loss_type']

    epochs = int(round(n_iters / (len ( train_set ) / batch_size)))
    learning_rate = run_params['lr']

    print (f"Start train model. Hidden size:{hidden_sizes}, batch_size:{batch_size}, n_iters:{n_iters}, epochs:{epochs}")
    use_pow = 'use_' if powers else 'no_'
    run_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}_v{model_n}.{s_model}.{use_pow}pow_epochs_{epochs}_batch_size_{batch_size}_lr_{learning_rate}"

    # Create model run dir and save run_params: # TODO: split part this to a function
    wdir = get_model_dirs ( run_name ,model_n,s_model)

    params_fname = os.path.join(wdir['main'],'run_params.json')
    run_params.update({'csv_path':os.path.join(os.getcwd(),csv_path), 'in_channels':in_channels})
    with open(params_fname,'w+') as f:
        json.dump(run_params,f)

    model = DefaultCNN(in_channels = in_channels,output_size = output_size,hidden_sizes = hidden_sizes).to(device)
    numParams = 0
    for parameter in model.parameters ( ) :
        if (parameter.requires_grad) :
            numParams += parameter.numel ( )
    print( f"Number of parameters in model: {numParams}" )


    # Step 3. Instantiate Loss Class
    if loss_type == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_type == 'MAELoss':
        criterion = nn.L1Loss()
    elif loss_type == 'MARELoss':
        criterion = MARELoss() # MeanAbsoluteRelativeError() #MARELoss() ##

    # Step 4. Instantiate Optimizer Class
    optimizer = torch.optim.Adam ( model.parameters ( ) , lr = learning_rate )

    # Step 5. Initiate dataloader
    train_loader = torch.utils.data.DataLoader (
        train_set , batch_size = batch_size , shuffle = True , num_workers = 7 )

    val_loader = torch.utils.data.DataLoader (
        val_set , batch_size = batch_size , shuffle = False , num_workers = 7 )

    writer = SummaryWriter ( wdir['run'])
    # Training loop
    best_loss = None
    best_epoch = None
    for epoch in range(1, epochs+1):
        model.train()   # Training mode
        running_loss = 0.0
        count_less = 0.0
        epoch_time = time.time ( )
        for i_batch , sample_batched in enumerate ( train_loader ) :
            # get inputs of x,y & send to device
            x = sample_batched [ 'x' ].to ( device )
            y = sample_batched [ 'y' ].to ( device )

            # forward + backward + optimize
            y_pred = model(x)   # Forward
            try:
                loss = criterion(y,y_pred)  # set loss calculation
            except Exception as e:
                print(f"Exception: {e}")
                print (f"Current sample: \n x = {x}\n y = {y}")
                print (f"Current prediction:\n y_pred = {y}")
                print(f"Current indexes = {sample_batched [ 'idx' ]}")
                print(f"Skipping the current batch")
                count_less+=1
                continue
            optimizer.zero_grad()       # zeroing parameters gradient
            loss.backward()             # backpropagation
            optimizer.step()            # update parameters

            # for statistics
            global_iter = len(train_loader)*(epoch-1)+i_batch
            writer.add_scalar ( f"{loss_type}/global_iter" , loss ,global_iter )
            running_loss += loss.data.item()

        # Normalizing loss by the total batches in train loader
        running_loss/= (len(train_loader)+count_less)
        writer.add_scalar ( f"{loss_type}/running_loss" , running_loss , epoch )

        # Calculate statistics for current model
        epoch_loss, feature_loss = update_stats ( writer , model , [ train_loader, val_loader ] , device , run_params , criterion , epoch )

        # Save best loss (and the epoch number) for train and validation
        if best_loss is None:
            best_loss = epoch_loss
            best_epoch = {'train': epoch, 'val':epoch}
            best_feture_loss = feature_loss
            best_feature_epoch={}
            for mode in ['train','val']:
                best_feature_epoch.update({mode:{}})
                for feature in run_params['Y_features']:
                    best_feature_epoch[mode].update({feature: epoch})
        else:
            for mode in ['train','val']:
                if best_loss[mode]> epoch_loss[mode]:
                    best_loss [ mode ] = epoch_loss[mode]
                    best_epoch[ mode ] = epoch
                for feature in run_params['Y_features']:
                    if best_feture_loss [ mode ][feature] > feature_loss [ mode ][feature]:
                        best_feture_loss [ mode ] [ feature ] = feature_loss [ mode ][feature]
                        best_feature_epoch [ mode ][feature] = epoch



        epoch_time = time.time ( ) - epoch_time

        # log statistics
        log = f"Epoch: {epoch} | Running {loss_type} Loss: {running_loss:.4f} |" \
              f" Train {loss_type}: {epoch_loss['train']:.3f} | Val {loss_type}: {epoch_loss['val']:.3f} | " \
              f" Epoch Time: {epoch_time:.2f} secs"
        print( log )

        # Save the model
        state = {
            'model_state_dict' : model.state_dict ( ) ,
            'optimizer_state_dict' : optimizer.state_dict() ,
            'learning_rate' : learning_rate,
            'batch_size':batch_size,
            'global_iter':global_iter,
            'epoch' : epoch ,
            'n_iters':n_iters,
            'in_channels' : in_channels,
            'output_size' : output_size
        }

        model_fname = os.path.join(wdir['checkpoints'], f"checkpoint_epoch_{epoch}-{epochs}.pth")
        torch.save ( state , model_fname )



        print ( f"Finished training epoch {epoch}, saved model to: {model_fname}" )
    write_hparams(writer,run_params, run_name='hparams',
                  cur_loss ={'loss':epoch_loss, 'epoch':epoch},
                  best_loss={'loss':best_loss,'epoch':best_epoch},
                  cur_loss_feature={'loss':feature_loss,'epoch':epoch},
                  best_loss_feature={'loss':best_feture_loss,'epoch':best_feature_epoch})
    writer.flush ( )


if __name__ == '__main__' :
    station_name = 'haifa'
    start_date = datetime ( 2017 , 9 , 1 )
    end_date = datetime ( 2017 , 10 , 31 )
    learning_rates = [1e-3, 0.5*1e-3, 1e-4]
    batch_sizes = [8]
    n_iters = 6000
    Y_features = [['r0' , 'r1'], ['r0','r1','LC'],['r0','r1','dr'], ['r0','r1','dr','LC'],['LC']]
    powers = [None,{'range_corr' : 0.5 , 'attbsc' : 0.5 , 'LC' : 0.5 ,
                    'LC_std' : 0.5 , 'r0' : 1 , 'r1' : 1, 'dr' : 1}]
    wavelengths = [ 355 , 532 , 1064 ]
    loss_types = ['MAELoss']  # 'MARELoss' 'MSELoss'
    hidden_sizes = [ 16 , 32 , 8 ] # TODO: add option - hidden_sizes = [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
    model_n = 1
    for loss_type in loss_types:
        for s_model,yf in enumerate(Y_features):
            if s_model<3:
                continue
            for lr in learning_rates:
                if s_model ==3 and lr>1e-4:
                    continue
                for pow in powers :
                    if s_model == 3 and pow==None:
                        continue
                    for batch_size in batch_sizes:
                        run_params = {'model':model_n, 's_model':s_model ,'hidden_sizes':hidden_sizes,
                                      'loss_type' :loss_type,
                                      'powers':pow,'Y_features': yf, 'wavelengths':wavelengths,
                                      'lr':lr, 'batch_size':batch_size, 'n_iters': n_iters}
                        main ( station_name , start_date , end_date, run_params )