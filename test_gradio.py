import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import gradio as gr

def lowlight(image_path):
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    # data_lowlight = data_lowlight.cuda().unsqueeze(0)
    data_lowlight = data_lowlight.to(device='cuda').unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch199.pth'))
    start = time.time()
    _,enhanced_image,_ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data','result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path


def enhanceImage(imPath):
    with torch.no_grad():
        result_path = lowlight(imPath)
    return result_path

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

html = "<html><h1> GENERATED OUTPUT </h1></html>"
html = (
    "<div style='max-width:100%; max-height:360px; overflow:auto'>"
    + html
    + "</div>"
)
gr_input = [gr.Image(type="filepath", shape=(512,512))]
gr_output = [gr.Image(type="filepath", shape=(512,512)),gr.Markdown(html)]



demo = gr.Interface(
    enhanceImage, 
    gr_input, 
    gr_output,
    theme='freddyaboulton/dracula_revamped',
    css=".gradio-container {background-color: plum}",
    # css="""
    # body{background-color : salmon}
    # """
    )
demo.launch()

# if __name__ == '__main__':
#  # test_images
#      with torch.no_grad():
#          filePath = 'data/test_data/'
#          # filePath = 'data/LOL/our485/low/'
    
#          file_list = os.listdir(filePath)

#          for file_name in file_list:
#              test_list = glob.glob(filePath+file_name+"/*") 
#              for image in test_list:
#                  # image = image
#                  print(image)
#                  lowlight(image)
# demo.launch()
        

