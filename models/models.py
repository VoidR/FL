import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    # if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        # init.kaiming_normal(m.weight)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, input_channels, num_classes):
        super(ResNet1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Linear(nn.Module):

    def __init__(self, num_in, num_out):
        super(Linear, self).__init__()
        self.fc = nn.Linear(num_in, num_out, bias=False)
        self.num_in, self.num_out = num_in, num_out
    def forward(self, x):
        self.y = self.fc(x)
        return self.y


class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # self.id_filter = torch.randn(out_channel,out_channel,1,1); self.id_filter[:, :, 0, 0] = torch.eye(out_channel)
        # print (self.id_filter)
        self.in_channel, self.out_channel = in_channel, out_channel
        


    def forward(self, x):
        self.x0 = x
        self.y = self.conv(x)
        # self.y0 = self.y
        # self.y = F.conv2d(self.y, self.id_filter, padding=0)
        return self.y

 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.cls_num = num_classes
        self.in_planes = 16

        self.conv1 = Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.y0 = out
        out = self.layer1(out)
        self.y1 = out
        out = self.layer2(out)
        self.y2 = out
        out = self.layer3(out)
        self.y3 = out
        out = F.avg_pool2d(out, out.size()[3])
        self.p = out
        out = out.view(out.size(0), -1)
        self.y_last = out
        out = self.linear(out)
        self.logits = out
        return out



def ResNet18(input_channels, num_classes):
    return ResNet1(BasicBlock1, [2,2,2,2], input_channels, num_classes)

def ResNet20(input_channels, num_classes):
    return ResNet(BasicBlock1, [3,3,3], input_channels, num_classes)

def ResNet32(input_channels, num_classes):
    return ResNet(BasicBlock1, [5,5,5], input_channels, num_classes)

def ResNet44(input_channels, num_classes):
    return ResNet(BasicBlock1, [7,7,7], input_channels, num_classes)

def ResNet56(input_channels, num_classes):
    return ResNet(BasicBlock1, [9,9,9], input_channels, num_classes)

def get_model(model_name, input_channels, num_classes):
    """
    根据模型名称返回相应的模型实例
    
    参数:
    - model_name: 模型名称 ('simple_cnn' 或 'resnet18')
    - input_channels: 输入通道数
    - num_classes: 类别数量
    
    返回:
    - 模型实例
    """
    if model_name == 'simple_cnn':
        return SimpleCNN(input_channels, num_classes)
    elif model_name == 'resnet18':
        return ResNet18(input_channels, num_classes)
    elif model_name == 'resnet20':
        return ResNet20(input_channels, num_classes)
    elif model_name == 'resnet32':
        return ResNet32(input_channels, num_classes)
    elif model_name == 'resnet44':
        return ResNet44(input_channels, num_classes)
    elif model_name == 'resnet56':
        return ResNet56(input_channels, num_classes)
    else:
        raise ValueError("不支持的模型名称")