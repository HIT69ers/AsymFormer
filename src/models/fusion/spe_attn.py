import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 
    

def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, N, groups, channels_per_group)

    # Transpose operation is not valid for 5D tensor, so we need to use permute
    x = x.permute(0, 1, 3, 2).contiguous()

    # flatten
    x = x.view(batchsize, N, -1)

    return x


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2):
        super(Cross_Atten_Lite_split, self).__init__()
        self.midc1 = torch.tensor(inc1 // 4)
        self.midc2 = torch.tensor(inc2 // 4)

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout = nn.Dropout(0.2)
        self._init_weight()

    def forward(self, x, x1, x2):
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)

        x1 = self.bn_x1(x1)
        x2 = self.bn_x2(x2)

        kq1 = self.kq1(x1.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq2 = self.kq2(x2.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = torch.split(kq, self.midc2, dim=2)

        v = self.v_conv(x.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        v1, v2 = torch.split(v, self.midc1, dim=2)

        mat = torch.matmul(q1, k1.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v1 = torch.matmul(mat, v1)

        mat = torch.matmul(q2, k2.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v2 = torch.matmul(mat, v2)

        v = torch.cat([v1, v2], dim=2).view(batch_size, h, w, -1)
        v = self.out_conv(v)
        v = v.permute(0, 3, 1, 2)
        v = self.bn_last(v)
        v = v + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SpatialAttention_new(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_new, self).__init__()
        self.inc = torch.tensor(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.fc_spatial = nn.Sequential(
        #     nn.Linear(in_channels, in_channels // reduction1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_channels // reduction1, in_channels, bias=False),
        # )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self.spe_attn = SAM()

        self._init_weight()

    def forward(self, x):

        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).view(b, c)

        y_channel = self.fc_channel(y_avg).view(b, c, 1, 1)
        y_channel = y_channel.sigmoid()

        output = x * y_channel
        output = self.spe_attn(output)

        return output + x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        channel = inc_rgb + inc_depth2

        self.fus_atten = SpatialAttention_new(in_channels=channel)
        self.conv1 = nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(inc_depth2)

        self.cross_atten = Cross_Atten_Lite_split(inc_depth2, inc_rgb)

    def forward(self, depth_out, rgb_out):
        fus_s = torch.cat([depth_out, rgb_out], dim=1)
        fus_s = self.fus_atten(fus_s)
        fus_s = self.conv1(fus_s)
        fus_s = self.bn(fus_s)

        fus_s = self.cross_atten(fus_s, depth_out, rgb_out)

        return fus_s