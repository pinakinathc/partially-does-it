from turtle import forward
import torch
import torchvision
import torch.nn as nn
import torchvision.models as backbone_
import encoding
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, arch='vgg-16'):
        super().__init__()
        if arch == 'vgg-16':
            self.encoder = VGG_Network()
        elif arch == 'resnet50':
            self.encoder = ResNet50_Network()
        elif arch == 'inception':
            self.encoder = Inception_Network()
        else:
            raise ValueError('arch parameter %s not found'%arch)
    
    def forward(self, input_):
        return self.encoder(input_)
        

class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

    def forward(self, input_):
        x = self.backbone(input_) # x: B x 512 x 7 x 7
        return x


class ResNet50_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())

    def forward(self, input_):
        x = self.backbone(input_)
        x = self.fc(x)
        return x


class Inception_Network(nn.Module):
    def __init__(self):
        super(Inception_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        #self.backbone.aux_logits = False
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        #self.linear_classification = nn.Linear(2048,125)

        self.head_layer = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(2048, 64),
            encoding.nn.Normalize())

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8        # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # # N x 2048 x 1 x 1
        # x = x.view(x.size(0), -1) #N x 2048
        # ##class_prediction = self.linear_classification(x) #N x 125
        # #embedding = F.normalize(x) #N x 2048
        # embedding = self.head_layer(x)
        # return embedding
        return x

    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False