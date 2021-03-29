import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam

from utils_.custom_losses import MARELoss


class DefaultCNN(LightningModule):

    def __init__(self, in_channels=2, output_size=3, hidden_sizes=[16, 32, 64], loss_type='MSELoss', learning_rate=1e-3):
        super().__init__()
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
            nn.Conv2d(in_channels=hidden_sizes[1], out_channels=hidden_sizes[2], kernel_size=3,
                      padding=1),
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
        if loss_type == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAELoss':
            self.criterion = nn.L1Loss()
        elif loss_type == 'MARELoss':
            self.criterion = MARELoss()  # MeanAbsoluteRelativeError() #MARELoss() ##

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # conv layers
        x = self.conv_layer(x.float()) # TODO show adi addition of .float

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
        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_pred = self(x)
        loss = self.criterion(y, y_pred)
        self.log("ptl/val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

