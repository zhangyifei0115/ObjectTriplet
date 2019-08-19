import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet101

class ObjectTripletModel(nn.Module):

    def __init__(self, embedding_size, num_classes, pretrained=True):
        super(ObjectTripletModel, self).__init__()

        self.model = resnet34(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(2048, self.embedding_size) # 256*256->131072
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(selfl, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1) #(120,120)->[,8192]
        # print('111111111111111111111')
        # print(x.shape)
        # print('11111111111111111111111')
        x = self.model.fc(x)
        # print('00000000000000000000000000')
        # print(x.shape)
        # print('00000000000000000000000000')

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)

        return res

