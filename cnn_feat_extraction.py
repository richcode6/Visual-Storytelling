#!/usr/bin/env python3

import os
import sys
import threading
import numpy as np
import yaml
import pickle
import pdb
import re
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision import transforms
from joblib import Parallel, delayed

import threading
import time
import multiprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_cnn_features_from_image(img, cnn_feat_video_filename):
    # Receives filename of downsampled video and of output path for features.
    # Extracts features in the given keyframe_interval. Saves features in pickled file.
    print("Processing {}".format(cnn_feat_video_filename))
    x = Variable(transform(img))
    x = x.unsqueeze(0)
    my_embedding = torch.zeros(1, 2048, 1, 1)

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    h = layer.register_forward_hook(copy_data)
    h_x = model(x)
    h.remove()
    z = my_embedding.data.numpy()
    z = z.squeeze(-1).squeeze(-1)
    np.savez(cnn_feat_video_filename, z)
    print("Saved {}".format(cnn_feat_video_filename))


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def get_cnn_features(fread):
    i = 0
    for image_name in fread:
        image_filename = os.path.join(image_folder_path, image_name)
        cnn_feat_filename = os.path.join(cnn_features_folderpath, image_name)
        try:
            img = Image.open(image_filename).convert('RGB')
            if not os.path.isfile(image_filename):
                print("{} File not found!".format(image_filename))
                continue
            if os.path.exists(cnn_feat_filename+".npz"):
                # print("{} Skipped".format(cnn_feat_filename))
                continue
            get_cnn_features_from_image(img,
                                          cnn_feat_filename)
        except:
            print("{} Exception".format(image_filename))
            continue
        i += 1
        if i%500==0:
            print("Processed {}".format(i))


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    image_folder_path = sys.argv[1]
    cnn_features_folderpath = sys.argv[2]

    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = [f for f in os.listdir(image_folder_path)] #if re.match(r'[0-9]+.*\.jpg', f)]
    print(len(fread))
    lines = chunkIt(fread, 1)
    print(len(lines))
    i = 0

    _transforms = list()
    _transforms.append(transforms.Resize((256, 256)))
    _transforms.append(transforms.CenterCrop(224))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)

    arch = "resnet50"
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    layer = model._modules.get('avgpool')

    # cnn.cuda()
    print("Initialized!")
    # Remove final classifier layer
    start = time.time()
    print("Time: {}".format(start))

    Parallel(n_jobs=16)(delayed(get_cnn_features)(line) for line in lines)
    # thread = [None for _ in range(len(lines))]
    # my_embedding = [None for _ in range(len(lines))]
    #
    # for i in range(0, len(thread)):
    #     print(len(lines[i]))
    #     thread[i] = threading.Thread(target=get_cnn_features,
    #                                args=(lines[i],))
    #
    # for i in range(0, len(thread)):
    #     thread[i].start()
    #
    # for i in range(0, len(thread)):
    #     thread[i].join()

    print("Time: {}".format(time.time()-start))