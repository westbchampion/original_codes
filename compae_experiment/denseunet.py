import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layer(x)

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.transp_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.transp_conv(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.layer(x)

class DenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, growth_rate=12, num_layers_per_block=4):
        super(DenseUNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers_per_block = num_layers_per_block

        self.init_conv = nn.Conv2d(in_channels, growth_rate * 2, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        self.skip_channels = []
        num_channels = growth_rate * 2

        for _ in range(4):
            block = DenseBlock(num_channels, growth_rate, num_layers_per_block)
            self.down_blocks.append(block)
            num_channels = num_channels + growth_rate * num_layers_per_block
            self.skip_channels.append(num_channels)
            trans_down = TransitionDown(num_channels)
            self.trans_downs.append(trans_down)

        # Bottleneck
        self.bottleneck = DenseBlock(num_channels, growth_rate, num_layers_per_block)
        num_channels = num_channels + growth_rate * num_layers_per_block

        # Decoder
        self.trans_ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.skip_channels = self.skip_channels[::-1]

        for idx in range(4):
            skip_channel = self.skip_channels[idx]
            trans_up = TransitionUp(num_channels, skip_channel)
            self.trans_ups.append(trans_up)

            num_channels = skip_channel + skip_channel  # Concatenate channels
            block = DenseBlock(num_channels, growth_rate, num_layers_per_block)
            self.up_blocks.append(block)
            num_channels = num_channels + growth_rate * num_layers_per_block

            # Transition Layer to reduce channels
            trans = TransitionLayer(num_channels, skip_channel)
            self.transitions.append(trans)
            num_channels = skip_channel  # After transition, channels reduced to skip_channel

        # Final Conv
        self.final_conv = nn.Conv2d(num_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)
        skip_connections = []

        # Encoder
        for block, trans_down in zip(self.down_blocks, self.trans_downs):
            x = block(x)
            skip_connections.append(x)
            x = trans_down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for trans_up, block, trans, skip in zip(self.trans_ups, self.up_blocks, self.transitions, reversed(skip_connections)):
            x = trans_up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = trans(x)  # Reduce channels

        x = self.final_conv(x)
        return x

# 示例用法：
if __name__ == "__main__":
    model = DenseUNet(in_channels=3, out_channels=1, growth_rate=12, num_layers_per_block=4)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)  # 输出应为 [1, 1, 256, 256]
