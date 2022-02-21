import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import Adam

from learning_lidar.learning_phase.learn_utils.custom_losses import MARELoss


class PowerLayer(nn.Module):
    def __init__(self, powers=[1, 1], do_opt_powers: bool = False):
        super(PowerLayer, self).__init__()
        self.powers = nn.Parameter(torch.tensor(powers))
        self.train_powers(do_opt_powers)

    def train_powers(self, do_opt_powers: bool = False):
        self.powers.requires_grad = do_opt_powers
        self.powers.retain_grad = do_opt_powers

    def forward(self, x):
        for c_i, p_i in enumerate(self.powers):
            x[:, c_i, :, :] = torch.pow(x[:, c_i, :, :], p_i)
        return x


class DefaultCNN(LightningModule):

    def __init__(self, in_channels, output_size, hidden_sizes, fc_size, loss_type, learning_rate, X_features_profiles,
                 powers, weight_decay=0, do_opt_powers: bool = False, conv_bias: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.eps = torch.tensor(np.finfo(float).eps)
        self.cov_bias = conv_bias
        X_features, profiles = map(list, zip(*X_features_profiles))
        self.x_powers = [powers[profile] for profile in profiles] if powers else None

        self.power_layer = PowerLayer(self.x_powers)
        self.conv_layer = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_sizes[0], kernel_size=(5, 3), padding=3,
                      bias=self.cov_bias),
            nn.BatchNorm2d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 2
            nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[1], kernel_size=3, padding=1,
                      bias=self.cov_bias),
            nn.BatchNorm2d(hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 3
            nn.Conv2d(in_channels=hidden_sizes[1], out_channels=hidden_sizes[2], kernel_size=3, padding=1,
                      bias=self.cov_bias),
            nn.BatchNorm2d(hidden_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),

            # Conv layer 4
            nn.Conv2d(in_channels=hidden_sizes[2], out_channels=hidden_sizes[3], kernel_size=3, padding=1,
                      bias=self.cov_bias),
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

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.float()
        if self.x_powers is not None:
            x = self.power_layer(x)

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
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
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
        # Log Y values for overfitt test
        if (self.trainer.overfit_batches >= 0) & (self.current_epoch % 10 == 0):
            for ind, (y_i, y_pred_i) in enumerate(zip(y,y_pred)):
                self.log(f"overfitt/y_{ind}/orig", y_i)
                self.log(f"overfitt/y_{ind}/pred", y_pred_i)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
