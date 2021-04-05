import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class inception_module(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool):
        super(inception_module, self).__init__()
        
        self.conv_1 = conv_1(in_dim, out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3)
        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5)
        self.max_3_1 = max_3_1(in_dim, pool)
        
    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        output = torch.cat([out_1, out_2, out_3, out_4], 1)
        return output
    
    # 인셉션 모듈을 위한 블럭 설정
    def conv_1(in_dim, out_dim): # 1 x 1 연산입니다.
        model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, 1), # k 와 s 를 1로 둡니다.
                nn.ReLU()
                )
        return model

    def conv_1_3(in_dim, mid_dim, out_dim): # 1 x 1, 3 x 3 연산입니다.
        model = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, 1, 1),
                nn.ReLU(),
                nn.Conv2d(mid_dim, out_dim, 3, 1, 1),
                # k = 3, s = 1, p = 1 로 둡니다. s = 1 이면, I - K + 2P + 1 이기 때문에 텐서 shape 이 동일합니다.
                nn.ReLU()
        )
        return model

    def conv_1_5(in_dim, mid_dim, out_dim):
        model = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, 1, 1),
                nn.ReLU(),
                nn.Conv2d(mid_dim, out_dim, 5, 1, 2),
                # 마찬가지로 k = 5, s = 1, p = 2 이면, I - 5 + 2*2 + 1 이기 때문에 텐서 shape 이 동일합니다.
                nn.ReLU()
        )
        return model

    def max_3_1(in_dim, out_dim):
        model = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                # MaxPool 은 k = 3, s = 1, p = 1 로 받았습니다. 텐서 shape 은 여전히 동일합니다.
                nn.Conv2d(in_dim, out_dim, 1, 1),
                nn.ReLU()
        )
        return model

class GoogLeNet(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(GoogLeNet, self).__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base_dim, base_dim*3, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer_2 = nn.Sequential(
            inception_module(base_dim*3, # in_dim
                             64, # out_dim_1
                             96, # mid_dim_3
                             128, # out_dim_3
                             16, # mid_dim_5
                             32, # out_dim_5
                             32 # pool
                            ),
            inception_module(base_dim*4, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer_3 = nn.Sequential(
            inception_module(480, 192, 96, 208, 16, 48, 64),
            inception_module(512, 160, 112, 224, 24, 64, 64),
            inception_module(512, 128, 128, 256, 24, 64, 64),
            inception_module(512, 112, 144, 288, 32, 64, 64),
            inception_module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer_4 = nn.Sequential(
            inception_module(832, 256, 160, 320, 32, 128, 128),
            inception_module(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1)
        )
        
        self.layer_5 = nn.Dropout2d(0.4)
        self.fc_layer = nn.Linear(1024, 1000)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        
        return out