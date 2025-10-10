import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/autonomousvision/gaussian-opacity-fields
def decouple_appearance(image, gaussians, view_idx):  #输入渲染的图像，高斯模型，渲染这张图片的相机id   将输入图像和高斯模型的外观特征结合，通过外观网络生成映射图像，并对原始图像进行变换
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)  #可训练的张量
    H, W = image.size(1), image.size(2)
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]#图像下采样原始分辨率的1/32

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]  #将下采样的图片与获取的外观嵌入图片进行合并
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()  #将合并后的图像和外观嵌入传递    生成与原始图像分辨率一致的映射图像
    transformed_image = mapping_image * image   #将映射图像与原图像进行逐元素相乘，从而得到变换后的图像

    return transformed_image, mapping_image


class UpsampleBlock(nn.Module):  #上采样   逐步提升分辨率，提取高分辨率特征
    def __init__(self, num_input_channels, num_output_channels):  #输入张量的通道数，输出张量的通道数
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)   #像素重排
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)  #上采样后的特征进行卷积
        self.relu = nn.ReLU()  #增加非线性特征
        
    def forward(self, x):  #前向传播
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x  #最终的上采样结果
    
class AppearanceNetwork(nn.Module):  #生成与输入图像相同分辨率的映射图像   从特征图生成一个映射图像，用于输入图像的外观进行调整
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)   #特征提取
        self.up1 = UpsampleBlock(256, 128)   #四个上采样逐步提高分辨率
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    H, W = 1200//32, 1600//32
    input_channels = 3 + 64
    output_channels = 3
    input = torch.randn(1, input_channels, H, W).cuda()
    model = AppearanceNetwork(input_channels, output_channels).cuda()
    
    output = model(input)
    print(output.shape)