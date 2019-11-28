import torch
from torch import nn
from torchvision import models


class FCN32(nn.Module):
    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def __init__(self,num_classes, pretrained=True):
        super(FCN32, self).__init__()        
        vgg = models.vgg16(pretrained=True)
        
        if pretrained:
            for param in vgg.features.parameters():
                param.requires_grad=False

        # adopted from https://github.com/shelhamer/fcn.berkeleyvision.org
        # padding of 100 is to make sure that the network output is aligned to the input size 
        # otherwise seq of conv and pooling operation results in an output which is much smaller than inp size        
        vgg.features[0].padding = (100,100)       

        self.features = nn.Sequential(*list(vgg.features.children()))
        
        self.fc = nn.Sequential(nn.Conv2d(512,4096,7),nn.ReLU(True),nn.Dropout(),nn.Conv2d(4096,4096,1), nn.ReLU(True),               nn.Dropout(),nn.Conv2d(4096,num_classes,1))
        self._init_weights()
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)

    def forward(self, inp):
        x = self.features(inp)
        x = self.fc(x)
        out = self.upsample(x) # (21,3,256,256)
        diff = out.size()[2] - inp.size()[2]
        if diff!=0:
            # slicing to get region of same dim as input (we use diff//2 as diff is total extra padding on both sides)
            return out[:, :, diff//2: (diff//2 + inp.size()[2]), diff//2: (diff//2 + inp.size()[3])] 
        else:
            return out
    

class FCN16(nn.Module):
    
    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  
                    
    def __init__(self,num_classes,pretrained=True):
        super(FCN16, self).__init__()
        
        vgg = models.vgg16(pretrained=True)
        
        if pretrained:
            for param in vgg.features.parameters():
                param.requires_grad=False
                
        # adopted from https://github.com/shelhamer/fcn.berkeleyvision.org
        # padding of 100 is to make sure that the network output is aligned to the input size 
        # otherwise seq of conv and pooling operation results in an output which is much smaller than inp size  
        vgg.features[0].padding = (100,100) 
        
        self.feat4 = nn.Sequential(*list(vgg.features.children())[:24])
        self.feat5 = nn.Sequential(*list(vgg.features.children())[24:])
        self.conv5 =  nn.Conv2d(512, num_classes, kernel_size=1)
        
        nn.init.xavier_uniform_(self.conv5.weight)
        self.conv5.bias.data.zero_()
        
        self.fc = nn.Sequential(nn.Conv2d(512,4096,7),nn.ReLU(True),nn.Dropout(),nn.Conv2d(4096,4096,1), nn.ReLU(True),               nn.Dropout(),nn.Conv2d(4096,num_classes,1))
        self._init_weights()
        self.upsample1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,stride=2, bias=False)
        
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)

    def forward(self, inp):
        conv4_out = self.feat4(inp)
        x = self.feat5(conv4_out)
        x = self.fc(x)
        conv5_out = self.conv5(conv4_out)
        up1 = self.upsample1(x)   
        if up1.shape != conv5_out.shape:
            diff = conv5_out.size()[2] - up1.size()[2]
            res = conv5_out[:, :, diff//2: (diff//2 + up1.size()[2]), diff//2: (diff//2 + up1.size()[3])] + up1
        else:
            res = conv5_out + up1
        out = self.upsample2(res)
        diff = out.size()[2] - inp.size()[2]
        if diff!=0:
            # slicing to get region of same dim as input (we use diff//2 as diff is total extra padding on both sides)
            return out[:, :, diff//2: (diff//2 + inp.size()[2]), diff//2: (diff//2 + inp.size()[3])] 
        else:
            return out