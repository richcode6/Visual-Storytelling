import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import torch.nn as nn
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from vist import VIST
from torchvision import models


class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, dii_path, vocab, validIds, transform=None, isVal=False):
        self.image_dir = image_dir

        self.vist = VIST(validIds, sis_path, dii_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform
        self.IsVal = isVal

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.cnn_in_feature = 1024
    def __getitem__(self, index):
        #print("Reached Normal get_item")
        imgFlag = False
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]
        images = []
        #Alldescs = []
        imgPath = "/home/ashwinsr/resnetFeatMod/" + str(story_id) + ".npz"
        if os.path.exists(imgPath):
            imgFlag = True
            images = torch.Tensor(np.load(imgPath)['arr_0'])
            #print("found and loaded ", images.shape)
            #exit(0)

        targets = []

        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            description = vist.imagesDesc[image_id]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            if not imgFlag:
                for image_format in image_formats:
                    try:
                        image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                    except Exception:
                        continue

                if self.transform is not None:
                    #print("transforming : ", os.path.join(self.image_dir, str(image_id) + image_format))#, " ", print(image.size()))
                    image = self.transform(image)

                images.append(image)
                '''desc = []
                desctokens = nltk.tokenize.word_tokenize(description.lower())
                desc.append(vocab('<start>'))
                desc.extend([vocab(token) for token in desctokens])
                desc.append(vocab('<end>'))
                Alldescs.append(torch.Tensor(desc))'''

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)
        '''
        for i in images:
            print(i.shape)
        stacked = 
        '''
        if not imgFlag:
            images = torch.stack(images)
        #print(type(images) ,images.shape,  " from get item")

        return images, targets, photo_sequence, album_ids, story_id, imgFlag#, Alldescs


    def __len__(self):
        return len(self.ids)

    def GetItem(self, index):
        print("Reached CAP GETITEM")
        exit(0)
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, image_id + image_format)).convert('RGB')
                    break
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        return images, targets, photo_sequence, album_ids

    def GetLength(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, photo_sequence_set, album_ids_set, storyID, preTrained = zip(*data)

    targets_set = []
    lengths_set = []
    desc_set = []
    desc_lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    '''for desc in AllDescs:
        lengths = [len(imageDesc) for imageDesc in desc]
        descriptions = torch.zeros(len(desc), max(lengths)).long()
        for i, imageDesc in enumerate(desc):
            end = lengths[i]
            descriptions[i, :end] = imageDesc[:end]
        desc_lengths_set.append(lengths)
        desc_set.append(descriptions)
    #print(type(image_stories), len(image_stories), image_stories[0])'''

    return image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set, storyID, preTrained#, desc_set, desc_lengths_set


def get_loader(root, sis_path, dii_path,  vocab, transform, batch_size, validIds, shuffle, num_workers, isVal=False):
    vist = VistDataset(image_dir=root, sis_path=sis_path, dii_path=dii_path, vocab=vocab, validIds=validIds, transform=transform, isVal=isVal)

    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader, vist.cnn_in_feature
