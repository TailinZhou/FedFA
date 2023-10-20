import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
 
import torchvision.models as models
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10


class Client_Model(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, args.dims_feature)#args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
         
        if self.name == 'mnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, args.dims_feature)#args.dims_feature=200
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'fmnist':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32*4*4, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=100
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
        
        if self.name == 'cifar10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'mixed_digit':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1024, 384) 
            self.fc2 = nn.Linear(384, 100) #args.dims_feature=100
            self.classifier = nn.Linear(100, self.n_cls)
            
        if self.name == 'cifar100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1024, 384) 
            self.fc2 = nn.Linear(384, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            
        if self.name == 'resnet18':
            resnet18 = ResNet18_cifar10()
            resnet18.fc = nn.Linear(512, 512) 

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18
            self.fc1 = nn.Linear(512, self.args.dims_feature) 
            self.classifier = nn.Linear(self.args.dims_feature, self.args.num_classes)
  
        if self.name == "Resnet50":#without replacement of bn layers by gn layers
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            self.fc1 = nn.Linear(2048, 512) 
            self.fc2 = nn.Linear(512, args.dims_feature) #args.dims_feature=192
            self.classifier = nn.Linear(args.dims_feature, self.args.num_classes)

    def forward(self, x):

        if self.name == 'Linear':
            x = self.fc(x)
        
        if self.name == 'mnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))  
            x = self.classifier(y_feature)
        
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
            
        if self.name == 'fmnist':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32*4*4)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = (self.fc2(x))
            x = self.classifier(y_feature)
        
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = self.fc2(x)
            x = self.classifier(y_feature)
            
        if self.name == 'mixed_digit':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)

            
        if self.name == 'cifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*32*32)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
        if self.name == "resnet18":
            x =  F.relu(self.model(x))
            y_feature = self.fc1(x)
            x = self.classifier(y_feature)

            
        if self.name == "Resnet50":
            x = self.features(x).squeeze().view(-1,2048)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            x = self.classifier(y_feature)
            
 

        return y_feature,x
    
class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        #self.bn1 = nn.BatchNorm2d(64)
        self.gn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.gn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        #self.bn3 = nn.BatchNorm2d(128)
        self.gn3 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    
        self.fc1 = nn.Linear(6272, 2048)
        #self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        #self.bn5 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        # x = F.relu(self.gn1(self.conv1(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.gn2(self.conv2(x)))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.gn3(self.conv3(x)))
        # x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # # x = self.bn4(x)
        # features = F.relu(self.fc2(x))
        # x = self.classifier(features)
        
        x = F.relu((self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu((self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu((self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        x = self.classifier(features)
        
        return features, x