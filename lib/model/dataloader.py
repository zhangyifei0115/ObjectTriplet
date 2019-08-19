import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
import copy
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from ..roi_data_layer.roidb import combine_roidb
from ..roi_data_layer.getAllRoI import getAllRoI

class TripletObjectDataset(Dataset):

    def __init__(self, AllRoI, class_21, roidb, num_triplets, transform = None):


        self.num_triplets = num_triplets
        self.transform = transform
        self.roidb = roidb
        self.training_triplets = self.generate_triplets(self.num_triplets, AllRoI, class_21, roidb)


    @staticmethod
    def generate_triplets(num_triplets, AllRoI, class_21, roidb):

        triplets = []

        # ##################################################################################
        # ##################################################################################
        # #dataset pascal_voc
        # AllRoI = dict()
        # class_21 = ['__background__',  # always index 0
        #                  'aeroplane', 'bicycle', 'bird', 'boat',
        #                  'bottle', 'bus', 'car', 'cat', 'chair',
        #                  'cow', 'diningtable', 'dog', 'horse',
        #                  'motorbike', 'person', 'pottedplant',
        #                  'sheep', 'sofa', 'train', 'tvmonitor']
        class_index = list(range(len(class_21)))
        # for i in class_index:
        #     AllRoI[class_21[i]] = []
        #
        # for img_index, img_roidb in enumerate(roidb): #
        #     if len(img_roidb['gt_classes']) >= 1:
        #         for box_index, perbox in enumerate(img_roidb['boxes']):
        #             AllRoI[class_21[img_roidb['gt_classes'][box_index]]].append([img_index, box_index])
        #
        #
        # ##################################################################################
        # ##################################################################################

        for _ in range(num_triplets): # random choose triplet
            class_index_t = copy.deepcopy(class_index) #deep copy
            # print('*(*(*(*(*(*(*(*(*(*(*(')
            # print(class_index_t)
            class_index_t.remove(0)
            anc_classes = np.random.choice(class_index_t) # choose anchor img
            class_index_t.remove(anc_classes)
            pos_classes = anc_classes # choose pos img

            neg_classes = np.random.choice(class_index_t) # choose neg img

            anchor_img_index = list(range(len(AllRoI[class_21[anc_classes]])))
            anchor_img_index_t = copy.deepcopy(anchor_img_index)
            anchor_img = np.random.choice(anchor_img_index) # box_index
            anchor_img_index_t.remove(anchor_img)
            pos_img = np.random.choice(anchor_img_index_t) # box_index


            neg_img_index = list(range(len(AllRoI[class_21[neg_classes]])))
            neg_img = np.random.choice(neg_img_index) # box_index

            anc_img_index, anc_box_index = AllRoI[class_21[anc_classes]][anchor_img]  # index
            anc_triplet = [class_21[anc_classes], anc_img_index, anc_box_index, roidb[anc_img_index]['flipped']]

            pos_img_index, pos_box_index = AllRoI[class_21[pos_classes]][pos_img]  # index
            pos_triplet = [class_21[pos_classes], pos_img_index, pos_box_index, roidb[pos_img_index]['flipped']]

            neg_img_index, neg_box_index = AllRoI[class_21[neg_classes]][neg_img]  # index
            neg_triplet = [class_21[neg_classes], neg_img_index, neg_box_index, roidb[neg_img_index]['flipped']]

            anc_box = roidb[anc_triplet[1]]['boxes'][anc_triplet[2]]
            pos_box = roidb[pos_triplet[1]]['boxes'][pos_triplet[2]]
            neg_box = roidb[neg_triplet[1]]['boxes'][neg_triplet[2]]

            if anc_box[3]-anc_box[1]<=0 or anc_box[2]-anc_box[0]<=0 or pos_box[3]-pos_box[1]<=0 \
                    or pos_box[2]-pos_box[0]<=0 or neg_box[3]-neg_box[1]<=0 or neg_box[2]-neg_box[0]<=0:
                continue

            triplets.append([anc_triplet, pos_triplet, neg_triplet])

        return triplets

    def __getitem__(self, idx):
        anc_triplet, pos_triplet, neg_triplet = self.training_triplets[idx]

        # print('#-#-#-#######################')
        # print(self.roidb)
        # # print(self.imdb.roidb[anc_triplet[1]])
        # print('-#-#-#-#-#-#-#-#-#')
        anc_img_path = self.roidb[anc_triplet[1]]['image'] # path
        # print('anc_img --- ' + anc_img)
        pos_img_path = self.roidb[pos_triplet[1]]['image']
        neg_img_path = self.roidb[neg_triplet[1]]['image']

        # print('anc_img： ' + str(anc_img_path))
        anc_img = io.imread(anc_img_path)
        # print('pos_img： ' + str(pos_img_path))
        pos_img = io.imread(pos_img_path)
        # print('neg_img： ' + str(neg_img_path))
        neg_img = io.imread(neg_img_path)

        if anc_triplet[3]: #fliped
            anc_img = np.flip(anc_img, 1)
        if pos_triplet[3]: #fliped
            pos_img = np.flip(pos_img, 1)
        if neg_triplet[3]: #fliped
            neg_img = np.flip(neg_img, 1)


        anc_box = self.roidb[anc_triplet[1]]['boxes'][anc_triplet[2]]
        pos_box = self.roidb[pos_triplet[1]]['boxes'][pos_triplet[2]]
        neg_box = self.roidb[neg_triplet[1]]['boxes'][neg_triplet[2]]

        # print('anc')
        # print(anc_box)
        # print(anc_img.shape)
        # print('pos')
        # print(pos_box)
        # print(pos_img.shape)
        # print('neg')
        # print(neg_box)
        # print(neg_img.shape)
        if len(anc_img.shape) < 3:
            # print(80 * 'a')
            # print('anc_img： ' + str(anc_img_path))
            # print(anc_box)
            anc_img = np.array([anc_img, anc_img, anc_img])
            anc_img = anc_img.transpose(1, 2, 0)
        if len(pos_img.shape) < 3:
            # print(80 * 'p')
            # print('pos_img： ' + str(pos_img_path))
            # print(pos_box)
            pos_img = np.array([pos_img, pos_img, pos_img])
            pos_img = pos_img.transpose(1, 2, 0)
        if len(neg_img.shape) < 3:
            # print(80 * 'n')
            # print('neg_img： ' + str(neg_img_path))
            # print(neg_box)
            neg_img = np.array([neg_img, neg_img, neg_img])
            neg_img = neg_img.transpose(1, 2, 0)

        anc_img = anc_img[anc_box[1]:anc_box[3], anc_box[0]:anc_box[2], :]
        pos_img = pos_img[pos_box[1]:pos_box[3], pos_box[0]:pos_box[2], :]
        neg_img = neg_img[neg_box[1]:neg_box[3], neg_box[0]:neg_box[2], :]

        # cv2.imwrite('./data_test/aaaa.png', anc_img)
        # breakk = 3/0

        # print('anc')
        # print(anc_box)
        # print(anc_img.shape)
        # print('pos')
        # print(pos_box)
        # print(pos_img.shape)
        # print('neg')
        # print(neg_box)
        # print(neg_img.shape)
        # if anc_img.shape[0] <= 0  or anc_img.shape[1] <= 0:
        #     print(40 * 'a1')
        #     print('anc_img： ' + str(anc_img_path))
        #     print(anc_box)
        # if pos_img.shape[0] <= 0  or pos_img.shape[1] <= 0:
        #     print(40 * 'p1')
        #     print('pos_img： ' + str(pos_img_path))
        #     print(pos_box)
        # if neg_img.shape[0] <= 0  or neg_img.shape[1] <= 0:
        #     print(40 * 'n1')
        #     print('neg_img： ' + str(neg_img_path))
        #     print(neg_box)

        #resize
        # anc_img = cv2.resize(anc_img, (64, 64))
        # pos_img = cv2.resize(pos_img, (64, 64))
        # neg_img = cv2.resize(neg_img, (64, 64))
        # print(anc_img)
        #print(anc_img.shape) #(64, 64, 3)
        # cv2.imwrite('./data_test/aaaa.png', anc_img)

        sample = {'anc_img': [anc_img, anc_triplet[0]], 'pos_img': [pos_img, anc_triplet[0]], 'neg_img': [neg_img, neg_triplet[0]]}
        # sample_classes = {'anc_img': anc_triplet[0], 'pos_img': pos_triplet[0], 'neg_img': neg_triplet[0]}

        if self.transform:
            sample['anc_img'][0] = self.transform(sample['anc_img'][0])
            sample['pos_img'][0] = self.transform(sample['pos_img'][0])
            sample['neg_img'][0] = self.transform(sample['neg_img'][0])

        return sample


    def __len__(self):

        return len(self.training_triplets)



def get_dataloader(imdb_name, imdbval_name, num_train_triplets, num_valid_triplets, batch_size, num_works):
    #my get data
    # imdb, roidb, ratio_list, ratio_index = combine_roidb(imdb_name)
    AllRoI, class_21, roidb = getAllRoI(imdb_name)
    AllRoI_val, class_21_val, roidb_val = getAllRoI(imdbval_name)

    # rootdir = roidb[0]['image'][:-11]##########/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}

    object_dataset = {
        'train' : TripletObjectDataset(
            AllRoI = AllRoI,
            class_21 = class_21,
            roidb = roidb,
            num_triplets = num_train_triplets,
            transform = data_transforms['train']
        ),
        'valid' : TripletObjectDataset(
            AllRoI=AllRoI_val,
            class_21=class_21_val,
            roidb=roidb_val,
            num_triplets = num_valid_triplets,
            transform=data_transforms['valid']
        )
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(object_dataset[x], batch_size = batch_size, shuffle = True, num_workers = num_works)
        for x in ['train', 'valid']
    }

    data_size = {x: len(object_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size

def get_test_dataloader(imdb_name, num_test_triplets, batch_size, num_works):
    #my get data
    # imdb, roidb, ratio_list, ratio_index = combine_roidb(imdb_name)
    AllRoI, class_21, roidb = getAllRoI(imdb_name)

    # rootdir = roidb[0]['image'][:-11]##########/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}


    object_dataset = {
        'test' : TripletObjectDataset(
            AllRoI=AllRoI,
            class_21=class_21,
            roidb=roidb,
            num_triplets = num_test_triplets,
            #transform=data_transforms['test']
        )
    }
    # print(object_dataset['test'][0]['anc_img'])
    # cv2.imwrite('./data_test/aaaa.png', object_dataset['test'][0]['anc_img'])

    dataloaders = {
        'test': torch.utils.data.DataLoader(object_dataset['test'], batch_size = batch_size, shuffle = False, num_workers = num_works)
    }

    data_size = {'test': len(object_dataset['test'])}

    #return all triplet

    return object_dataset['test']
