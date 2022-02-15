import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import Adam

from learning_lidar.learning_phase.learn_utils.custom_losses import MARELoss


class DefaultCNN(LightningModule):

    def __init__(self, in_channels, output_size, hidden_sizes, fc_size, loss_type, learning_rate, X_features_profiles,
                 powers, do_opt_powers: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.eps = torch.tensor(np.finfo(float).eps)
        X_features, profiles = map(list, zip(*X_features_profiles))
        self.x_powers = nn.Parameter(torch.tensor([powers[profile] for profile in profiles])) if powers else None
        self.train_powers(do_opt_powers)

        self.conv_layer = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_sizes[0], kernel_size=(5, 3), padding=3),
            nn.BatchNorm2d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 2
            nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 3
            nn.Conv2d(in_channels=hidden_sizes[1], out_channels=hidden_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 4
            nn.Conv2d(in_channels=hidden_sizes[2], out_channels=hidden_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_sizes[3]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
        )
        # TODO: calc the first FC layer automatically (not hard coded), based on prev layer dimensions.
        fc_2layer = nn.Sequential(
            nn.Linear(4 * 8 * hidden_sizes[3], fc_size[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(fc_size[0], output_size),
            nn.ReLU(inplace=True))

        fc_1layer = nn.Sequential(
            nn.Linear(4 * 8 * hidden_sizes[3], output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.fc_layer = fc_1layer if fc_size[0] == 1 else fc_2layer

        # Step 3. Instantiate Loss Class
        self.loss_type = loss_type
        if loss_type == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAELoss':
            self.criterion = nn.L1Loss()
        elif loss_type == 'MARELoss':
            self.criterion = MARELoss()
        elif loss_type == 'MAESmooth':
            self.criterion = nn.SmoothL1Loss()

        # Step 4. Instantiate Relative Loss - for accuracy
        self.rel_loss_type = 'MARELoss'
        self.rel_loss = MARELoss()

    # step 5. Set Regularize Loss
    def regulizer_weights(self, alpha=0.01, ord=1):
        params = torch.cat([p[1].view(-1) for p in self.named_parameters()])
        return alpha * torch.linalg.norm(params, ord=ord)

    def train_powers(self, do_opt_powers: bool = False):
        if self.x_powers is not None:
            self.x_powers.requires_grad = do_opt_powers
        else:
            pass

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.float()
        if self.x_powers is not None:
            # https://github.com/torch/nn/blob/872682558c48ee661ebff693aa5a41fcdefa7873/Power.lua
            for c_i in range(channels):
                x[:, c_i, :, :] = (x[:, c_i, :, :] + self.eps) ** self.x_powers[c_i]

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(batch_size, -1)

        # fc layer
        out = self.fc_layer(x)
        return out

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        # Uncomment if you wish to stop training the powers at some point of training process
        # cond_opt_pow = (self.current_epoch <= self.trainer.max_epochs)
        # self.train_powers(do_opt_powers = cond_opt_pow)
        y_pred = self(x)
        loss = self.criterion(y, y_pred)  # + self.regulizer_weights()
        if torch.isnan(loss):
            raise ValueError('Val loss is NaN!')
        self.log(f"loss/{self.loss_type}_train", loss)
        rel_loss = self.rel_loss(y, y_pred)
        self.log(f"rel_loss/{self.rel_loss_type}_train", rel_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        self.log(f"loss/{self.loss_type}_val", loss)
        # for feature_num, feature in enumerate(features):
        #     self.log(f"{"MARE_{feature_num}_val", loss_mare)
        rel_loss = self.rel_loss(y, y_pred)
        self.log(f"rel_loss/{self.rel_loss_type}_val", rel_loss)
        for c_i in range(x.size()[1]):
            self.log(f"gamma_x/channel_{c_i}", self.x_powers[c_i])
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
