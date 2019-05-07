import torch
import torch.nn as nn
import torchvision

torchvision.models.inception_v3(pretrained=True)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ImageEncoder(nn.Module):

    def __init__(self, hidden_dim, feature_extracting):
        super().__init__()
        self.cnn = torchvision.models.inception_v3(pretrained=True)
        layers = [self.cnn.Conv2d_1a_3x3, self.cnn.Conv2d_2a_3x3, self.cnn.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),\
                self.cnn.Conv2d_3b_1x1, self.cnn.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2),\
                self.cnn.Mixed_5b, self.cnn.Mixed_5c, self.cnn.Mixed_5d, self.cnn.Mixed_6a,\
                self.cnn.Mixed_6b, self.cnn.Mixed_6c, self.cnn.Mixed_6d, self.cnn.Mixed_6e]
        self.featrues = nn.Sequential(*layers)
        if feature_extracting:
            set_parameter_requires_grad(self.cnn, feature_extracting)
            set_parameter_requires_grad(self.featrues, feature_extracting)
        self.projection = nn.Linear(768, hidden_dim, bias=False)
        
    def forward(self, x):
        x = self.featrues(x)
        h, w = x.size(2), x.size(3)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)
        return x