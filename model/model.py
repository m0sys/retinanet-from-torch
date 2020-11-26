import torch.nn as nn
from torchvision.models import resnet34
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResnetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet34()
        in_feats = self.model.fc.in_features

        self.model.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)