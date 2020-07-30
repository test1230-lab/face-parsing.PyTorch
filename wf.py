#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
#im an idiot
print("wf mask generator")
value = input("test mode? y/n/? ")
if value == '?':
	print("outputs semi-transparent mask on the image. Example: https://imgur.com/a/bPbKbvF")
	value = input("test mode? y/n/? ")
if value == 'y':
	a = (0.6)
	b=(0.1)
	c=(1)
if value == 'n':
	a = (0)
	b=(1)
	c= (0)
print(a)
print(b)
print(c)

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 0

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), (a), vis_parsing_anno_color, (b), (c))

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    evaluate(dspth='data/aligned', cp='79999_iter.pth')
