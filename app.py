import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch 
import torch.nn as nn
import os


st.title('Chào mừng bạn đến với website dự đoán các hiện tượng thời tiết')
st.header('Điều bạn cần làm là cung cấp 1 bức ảnh cho chúng tôi')

def model_resnet():
    class Block(nn.Module):
        def __init__ (self,in_channels, out_channels, stride = 1):
            super(Block,self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels,out_channels,
                kernel_size = 3, stride = stride, padding =1
            )
            self.batch_norm1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(
                out_channels, out_channels,
                kernel_size = 3 , stride = 1 , padding =1
            )

            self.batch_norm2 = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential()

            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels , out_channels,
                        kernel_size =1 , stride = stride
                    ),
                    nn.BatchNorm2d(out_channels)
                )
        def forward(self,x):
            res = x.clone()
            x= self.conv1(x)
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.relu(x)
            x = x+ self.downsample(res)
            x = self.relu(x)
            return x
    class Resnet(nn.Module):
        def __init__ (self, residual_block, n_block, n_class):
            super(Resnet,self).__init__()
            self.conv1 = nn.Conv2d(3,64,
                    kernel_size = 7, stride = 2 , padding = 3
                                )
            self.batch_norm1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(
                kernel_size = 3, stride = 2, padding = 1
            )
            self.conv2 = self.create_layer(residual_block , 64 , 64,n_block[0] , 1 )
            self.conv3 = self.create_layer(residual_block , 64,128,n_block[1],2)
            self.conv4 = self.create_layer(residual_block , 128,256,n_block[2],2)
            self.conv5 = self.create_layer(residual_block , 256,512,n_block[3],2)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.flaten  = nn.Flatten()
            self.fc = nn.Linear(512,n_class)
        def create_layer(self,residual_block, in_channels, out_channels , n_block ,stride):
            blocks = []
            frist_block = residual_block(in_channels , out_channels, stride)
            blocks.append(frist_block)

            for i in range(1,n_block):
                block = residual_block(out_channels,out_channels, stride)
                blocks.append(block)
                layer = nn.Sequential(*blocks)
            return layer

        def forward(self,x):
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.avgpool(x)
            x = self.flaten(x)
            x = self.fc(x)
            return x
    n_block = [2,2,2,2]
    n_class = 11
    model = Resnet(
        Block,
        n_block,
        n_class
    )
    return model


model = model_resnet()
model.load_state_dict(torch.load('resnet_model.pth'))
model.eval()


def class_weather():
    root_dir = '/home/phan/final_python/weather-dataset/dataset'
    classes = {
    label_idx: class_name \
        for label_idx, class_name in enumerate(
            sorted(os. listdir(root_dir))
        )
    }
    return classes

classes = class_weather()



st.markdown("<h3>Chọn FILE</h3>", unsafe_allow_html=True)
image= st.file_uploader('', type=['jpg', 'jpeg', 'png'])
if image is  not None:
   
    image = Image.open(image)
    image = image.convert('RGB')
    st.image(image, caption="Ảnh cần dự đoán")
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image)

    if st.button('Dự đoán'):
        with torch.no_grad():
            batch = torch.unsqueeze(image_tensor,0)
            out = model(batch)

        _, predicted = torch.max(out, 1)
        index = predicted.item()
        result = classes[index]
        st.markdown("<h3>Kết quả dự đoán:</h3>", unsafe_allow_html=True)
        
        translate = {   0: "sương",
                        1: "sương mù",
                        2: "sương giá",
                        3: "băng",
                        4: "mưa đá",
                        5: "sấm sét",
                        6: "mưa",
                        7: "cầu vồng",
                        8: "băng tuyết",
                        9: "bão cát",
                        10: "tuyết"}
        
        st.markdown(f"<h2 style='color: green;'>{result} - {translate[index]}</h2>", unsafe_allow_html=True)

        