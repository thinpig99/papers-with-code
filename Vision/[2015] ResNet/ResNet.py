import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# bottleNeck block

class BottleNeck(nn.Module):
    def __init__(self, in_dim: int,
                 mid_dim: int,
                 out_dim: int,
                 act_fn,
                 down: bool = False # down 인수를 True로 넣으면 Down Sampling 진행.
                ) -> nn.Module:
        super(BottleNeck, self).__init__()
        self.act_fn = act_fn
        self.down = down
        
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn, 2),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn)
            )
            self.downsample = nn.Conv2d(in_dim, out_dim, 1, 2)
            
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn)
            )
            self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1) # 1 x 1 kernel 을 활용해 dim 을 맞춰준다.
    
    def forward(self, x):
        if self.down: # down sampling
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample # concat
        else: # dim equalizing
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x # concat
        return out
    
    # basic conv block
    def conv_block_1(in_dim: int, out_dim: int, act_fn, stride:int = 1) -> None:
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
            act_fn
        )
        return model

    def conv_block_3(in_dim: int, out_dim: int, act_fn) -> None:
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            act_fn
        )
        return model
    
    
# ResNet
class ResNet(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3), # in, out, k, s, p
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1) # k, s, p
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn, down=True)
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn, down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn, down=True)
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn)
        )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc_layer = nn.Linear(base_dim*32, num_classes)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out