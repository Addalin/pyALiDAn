import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import Adam

from learning_lidar.learning_phase.utils_.custom_losses import MARELoss


class DefaultCNN(LightningModule):

    def __init__(self, in_channels, output_size, hidden_sizes, loss_type, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

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

            # Conv layer 2
            nn.Conv2d(in_channels=hidden_sizes[1], out_channels=hidden_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_sizes[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 32 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_size),
            nn.ReLU(inplace=True)
        )

        # Step 3. Instantiate Loss Class
        self.loss_type = loss_type
        if loss_type == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAELoss':
            self.criterion = nn.L1Loss()
        elif loss_type == 'MARELoss':
            self.criterion = MARELoss()

        # Step 4. Instantiate Relative Loss - for accuracy
        self.rel_loss_type = 'MARELoss'
        self.rel_loss = MARELoss()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # conv layers
        x = self.conv_layer(x.float())

        # flatten
        x = x.view(batch_size, -1)

        # fc layer # TODO: add option to change in_channels to fc_layer (when changing hidden sizes of CNN)
        out = self.fc_layer(x)

        return out

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        if torch.isnan(loss):
            raise ValueError('Val loss is NaN!')
        self.log(f"{self.loss_type}_train", loss)
        rel_loss = self.rel_loss(y, y_pred)
        self.log(f"{self.rel_loss_type}_train", rel_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        self.log(f"{self.loss_type}_val", loss)
        # for feature_num, feature in enumerate(features):
        #     self.log(f"{"MARE_{feature_num}_val", loss_mare)
        rel_loss = self.rel_loss(y, y_pred)
        self.log(f"{self.rel_loss_type}_val", rel_loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
