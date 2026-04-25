import numpy as np
from scipy.fftpack import fft, dct
import torch
import torch.nn as nn
import torch.nn.functional as F


def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    # print('height:{}'.format(height))
    N = 8
    step = int(height/N)  # 28

    dct_img = np.zeros((1, N*N, step*step, 1),
                       dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N*N, step*step, 1))
    # print('dct_img:{}'.format(dct_img.shape))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(
                img[:, row:(row+step), col:(col+step)], dtype=np.float32)
            # print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step*step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    # for i in range(64):
    # [batch_size,64, 784,1]
    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DctCNN(nn.Module):
    """
    Input:  dct_img [B, 64, 250]
    Output: feature [B, 4096]
    """

    def __init__(
        self,
        model_dim=256,
        dropout=0.5,
        in_channel=64,
        branch1_channels=[64],
        branch2_channels=[48, 64],
        branch3_channels=[64, 96, 96],
        branch4_channels=[32],
        out_channels=64,
    ):
        super().__init__()

        # Input after unsqueeze: [B, 1, 64, 250]
        self.stem = nn.Sequential(
            ConvBNReLU(1, 32, kernel_size=(1, 3)),
            ConvBNReLU(32, 64, kernel_size=(1, 3)),
            ConvBNReLU(64, 128, kernel_size=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # Inception-like branches
        self.branch1 = ConvBNReLU(128, branch1_channels[0], kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            ConvBNReLU(128, branch2_channels[0], kernel_size=(1, 1)),
            ConvBNReLU(branch2_channels[0], branch2_channels[1], kernel_size=(
                1, 3), padding=(0, 1)),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(128, branch3_channels[0], kernel_size=(1, 1)),
            ConvBNReLU(branch3_channels[0], branch3_channels[1], kernel_size=(
                1, 3), padding=(0, 1)),
            ConvBNReLU(branch3_channels[1], branch3_channels[2], kernel_size=(
                1, 3), padding=(0, 1)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBNReLU(128, branch4_channels[0], kernel_size=(1, 1)),
        )

        inception_out = (
            branch1_channels[0]
            + branch2_channels[-1]
            + branch3_channels[-1]
            + branch4_channels[0]
        )

        self.final = nn.Sequential(
            ConvBNReLU(inception_out, out_channels, kernel_size=(1, 1)),
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Expected x: [B, 64, 250]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, 64, 250]

        x = self.stem(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.final(x)  # [B, 4096]

        return x


class multimodal_attention(nn.Module):
    def __init__(self, attention_dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))

        if scale is not None:
            attention = attention * scale

        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, float("-inf"))

        attention = self.softmax(attention)
        attention = self.dropout(attention)

        return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dropout=0.5):
        super().__init__()

        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)

        self.attention = multimodal_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        query/key/value: [B, D]
        return: [B, D]
        """
        residual = query

        query = query.unsqueeze(1)  # [B, 1, D]
        key = key.unsqueeze(1)      # [B, 1, D]
        value = value.unsqueeze(1)  # [B, 1, D]

        batch_size = query.size(0)

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        q = q.view(batch_size, 1, self.num_heads,
                   self.dim_per_head).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads,
                   self.dim_per_head).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads,
                   self.dim_per_head).transpose(1, 2)

        scale = self.dim_per_head ** -0.5

        context = self.attention(q, k, v, scale, attn_mask)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, 1, self.model_dim)

        output = self.linear_final(context).squeeze(1)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=256, ffn_dim=2048, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = self.layer_norm(residual + x)

        return x


class multimodal_fusion_layer(nn.Module):
    """
    MCAN-style bidirectional co-attention fusion layer.

    input:
        image_output: [B, D]
        text_output:  [B, D]

    output:
        fused output: [B, D]
    """

    def __init__(self, model_dim=256, num_heads=8, ffn_dim=2048, dropout=0.5):
        super().__init__()

        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward_1 = PositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)

        self.fusion_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):
        # image attends to text
        output_1 = self.attention_1(
            query=image_output,
            key=text_output,
            value=text_output,
            attn_mask=attn_mask
        )

        # text attends to image
        output_2 = self.attention_2(
            query=text_output,
            key=image_output,
            value=image_output,
            attn_mask=attn_mask
        )

        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)

        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output
