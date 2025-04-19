import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# for other networks
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class CAN(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SAN(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAN, self).__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class BAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = CAN(channel)
        self.spatial_attn = SAN()

    def forward(self, x):
        attn = x + self.channel_attn(x)
        attn = x + self.spatial_attn(attn)
        return attn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        ide = self.down(x)
        out += ide
        out = self.relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, channels, num_blocks):
        super(DenseBlock, self).__init__()
        self.conv = ResidualBlock(channels, channels)
        self.blocks = []
        for _ in range(1, num_blocks):
            self.blocks.append(ResidualBlock(channels, channels))

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        res = self.conv(x)
        for block in self.blocks:
            res = block(res)
        return res

class UpScaleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, scale_factor=2):
        super(UpScaleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups  = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False) # nearest
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ups(x)
        x = self.silu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Down, self).__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.act_fnc(self.norm1(x))
        s = self.conv3(x)
        x = self.act_fnc(self.norm2(self.conv1(x)))
        x = self.conv2(x)
        return x + s

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(Up, self).__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="bicubic")
        self.act_f = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.up_nn(self.act_f(self.norm1(x)))
        s = self.conv3(x)
        x = self.act_f(self.norm2(self.conv1(x)))
        x = self.conv2(x)
        return x + s

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        rx = x
        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Adjust x size to match skip connection
            if x.shape != skip_connection.shape:
                x = adjust_size(x, skip_connection)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)


        x = self.final_conv(x).float()
        return x + rx

def adjust(x1, x2):
    x1 = F.interpolate(x2, size=x2.shape[2:], mode='bicubic', align_corners=False)
    return x1

class Generator(nn.Module):
    def __init__(self, in_channels=3, dim=64, scale_factor=4, patch_size=16):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.SiLU(inplace=True)
        )

        self.ds1 = DenseBlock(dim, 4)
        self.at1 = BAM(dim)

        # ViT-like Transformer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(inplace=True)
        )
        # image shape is (B, C, H/p, W/p)
        self.unet1 = UNet(dim, dim)

        self.ds2 = DenseBlock(dim, 3)
        self.at2 = BAM(dim)

        # Attention mechanisms
        self.up1 = nn.Sequential(
            UpScaleBlock(dim),
            BAM(dim)
        )
        self.ds3 = DenseBlock(dim, 3)
        self.at3 = BAM(dim)

        self.up2 = nn.Sequential(
            UpScaleBlock(dim),
            BAM(dim)
        )
        self.ds4 = DenseBlock(dim, 3)
        self.at4 = BAM(dim)

        self.unet2 = UNet(dim, dim)

        # Final layers
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, in_channels, kernel_size=1, padding=0),
        )
        self.scale_factor = scale_factor
        initialize_weights(self, scale=0.1)

    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.ds1(x)
        x = self.at1(x)
        
        x = self.patch_embed(x)
        x = x + self.unet1(x)

        x = self.ds2(x)
        x = self.at2(x)

        # Apply attention mechanisms
        x = self.up1(x)
        x = self.ds3(x)
        x = self.at3(x)
        x = self.up2(x)
        x = self.ds4(x)
        x = self.at4(x)
        x = self.unet2(x)

        # Final layer
        x = self.conv2(x)
        r = F.interpolate(r, scale_factor=0.25, mode='bicubic', align_corners=False)
        x = adjust(x, r)
        x = r + x

        return x.clamp(0, 1)