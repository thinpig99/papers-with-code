import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim * 7 * 7, 100),
            nn.ReLU(True),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Linear(20, num_classes)
        )
    def forward(x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
    def conv_2_block(in_dim, out_dim): # 2개의 합성곱 레이어를 만드는 콘브 블럭입니다.
        model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
        )
        return model

    def conv_3_block(in_dim, out_dim): # 3개의 합성곱 레이어를 만드는 콘브 블럭입니다.
        model = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
        )
        return model