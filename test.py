from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import os.path as osp
import numpy as np
import argparse
from skimage import io
import time
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from lib.model.tripletutils import PairwiseDistance, TripletLoss
from lib.model.triplet import ObjectTripletModel
from lib.model.dataloader import get_test_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Train Triplet to clustering object')
    parser.add_argument('--dataset', dest = 'dataset',
                        help = 'trainging dataset',
                        default = 'pascal_voc', type = str)
    parser.add_argument('--net', dest = 'net',
                        help = 'vgg16 or resnet34,res101',
                        default = 'vgg16', type = str)
    parser.add_argument('--learning-rate', dest = 'lr',
                        default = 0.001, type = float)
    parser.add_argument('--embedding-size', dest = 'embedding_size',
                        default = 128, type = int,
                        help = 'embedding size')
    parser.add_argument('--margin', dest = 'margin',
                        default = 0.5, type = float)
    parser.add_argument('--num-classes', dest = 'num_classes',
                        default = 20, type = int,
                        help='number of classes')
    parser.add_argument('--start-epoch', dest = 'start_epoch',
                        default = 0, type = int)
    parser.add_argument('--num-epochs', dest = 'num_epochs',
                        default = 200, type = int)
    parser.add_argument('--save-dir', dest = 'save_dir',
                        default = './models', type = str)
    parser.add_argument('--num-train-triplets', dest = 'num_train_triplets',
                        default = 1000, type = int)
    parser.add_argument('--num-valid-triplets', dest = 'num_valid_triplets',
                        default = 1000, type = int)
    parser.add_argument('--num-test-triplets', dest='num_test_triplets',
                        default=1000, type=int)
    parser.add_argument('--batch-size', dest = 'batch_size',
                        default = 32, type = int)
    parser.add_argument('--num_works', dest = 'num_works',
                        default = 4, type = int)


    args = parser.parse_args()

    return args


def getDist(img_path, boxes, model):
    # get anchor positive negtive image
    # need hole image + box

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64), interpolation=3), #transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    l2_dist = PairwiseDistance(2)

    anc_img_path = img_path[0]
    pos_img_path = img_path[1]
    neg_img_path = img_path[2]

    anc_box = boxes[0]
    pos_box = boxes[1]
    neg_box = boxes[2]

    anc_img = io.imread(anc_img_path)
    pos_img = io.imread(pos_img_path)
    neg_img = io.imread(neg_img_path)

    anc_img = anc_img[anc_box[1]:anc_box[3], anc_box[0]:anc_box[2], :]
    pos_img = pos_img[pos_box[1]:pos_box[3], pos_box[0]:pos_box[2], :]
    neg_img = neg_img[neg_box[1]:neg_box[3], neg_box[0]:neg_box[2], :]

    # cv2.imwrite('anc_img.png', anc_img)
    # cv2.imwrite('pos_img.png', pos_img)
    # cv2.imwrite('neg_img.png', neg_img)

    anc_img = data_transform(anc_img)
    pos_img = data_transform(pos_img)
    neg_img = data_transform(neg_img)

    anc_img = anc_img.unsqueeze(0)
    pos_img = pos_img.unsqueeze(0)
    neg_img = neg_img.unsqueeze(0)

    anc_img = anc_img.to(device).detach()
    pos_img = pos_img.to(device).detach()
    neg_img = neg_img.to(device).detach()

    anc_embed = model(anc_img)
    pos_embed = model(pos_img)
    neg_embed = model(neg_img)

    pos_dist = l2_dist(anc_embed, pos_embed)
    pos_dist = torch.mean(pos_dist)
    neg_dist = l2_dist(anc_embed, neg_embed)
    neg_dist = torch.mean(neg_dist)

    return pos_dist, neg_dist

def test(model, object_dataset, num_classes, num_test_triplets):
    #kmeans cluster

    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    l2_dist = PairwiseDistance(2)

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64), interpolation=3),  # transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    model.eval()

    Features = []
    Images = []
    GT_labels = []

    for index in tqdm(range(num_test_triplets)):
        # anc_img = batch_sample['anc_img'].to(device)
        # print(anc_img.shape) # [5, 3, 64, 64]
        # im = anc_img[0].permute(1,2,0).cpu().detach().numpy()
        # print(im.shape)
        # cv2.imwrite('./data_test/aaaa.png', im)
        # print(im)
        # print(anc_img[0].cpu().detach().numpy().transpose(2,1,0))
        # 3/0
        # pos_img = batch_sample['pos_img'].to(device)
        # neg_img = batch_sample['neg_img'].to(device)
        org_anc_img = object_dataset[index]['anc_img'][0]
        org_pos_img = object_dataset[index]['pos_img'][0]
        org_neg_img = object_dataset[index]['neg_img'][0]
        # cv2.imwrite('./data_test/aaaa.png', anc_img)

        anc_img_class = object_dataset[index]['anc_img'][1]
        pos_img_class = object_dataset[index]['pos_img'][1]
        neg_img_class = object_dataset[index]['neg_img'][1]

        anc_img = data_transform(org_anc_img)
        pos_img = data_transform(org_pos_img)
        neg_img = data_transform(org_neg_img)

        anc_img = anc_img.unsqueeze(0)
        pos_img = pos_img.unsqueeze(0)
        neg_img = neg_img.unsqueeze(0)

        anc_img = anc_img.to(device).detach()
        pos_img = pos_img.to(device).detach()
        neg_img = neg_img.to(device).detach()

        anc_embed = model(anc_img)
        # print(anc_embed)
        # print(anc_embed[0].cpu().detach().numpy())
        pos_embed = model(pos_img)
        neg_embed = model(neg_img)

        # for img in anc_img:
        # print(img.cpu().detach().numpy() for img in anc_img )

        # for img in anc_img:
        Images.append(np.array(org_anc_img))
        # print(org_anc_img.shape)
        # for img in pos_img:
        Images.append(np.array(org_pos_img))
        # for img in neg_img:
        Images.append(np.array(org_neg_img))

        # for feat in anc_embed:
        Features.append(anc_embed.cpu().detach().numpy().squeeze(0))
        # print(anc_embed.cpu().detach().numpy().shape)
        # for feat in pos_embed:
        Features.append(pos_embed.cpu().detach().numpy().squeeze(0))
        # for feat in neg_embed:
        Features.append(neg_embed.cpu().detach().numpy().squeeze(0))

        GT_labels.append(anc_img_class)
        GT_labels.append(pos_img_class)
        GT_labels.append(neg_img_class)

    Feature = np.array(Features)
    GT_labels = np.array(GT_labels)
    print(Feature.shape)
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(Feature)

    kmeans_label = kmeans.labels_ #aaray([0,0,1,2,3])

    cluster_result_path = './cluster_result/'
    # for i in range(num_classes):
    #     cluster_img_path = cluster_result_path + str(i) + '/'
    #     if not os.path.exists(cluster_img_path):
    #         os.makedirs(cluster_img_path)

    label_count = np.zeros(num_classes)
    # print(Images[0].transpose(2, 1, 0))
    # print(Images[0].transpose(2, 1, 0).shape)
    for i, v in enumerate(kmeans_label):
        cluster_img_path = cluster_result_path + str(v) + '/'
        if not os.path.exists(cluster_img_path):
            os.makedirs(cluster_img_path)
        cv2.imwrite(cluster_img_path + '{}.png'.format(label_count[v]), Images[i])
        label_count[v] += 1





def mymain():

    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2014_minival"

    l2_dist = PairwiseDistance(2)
    model = ObjectTripletModel(embedding_size=args.embedding_size, num_classes=args.num_classes).to(device)
    if args.start_epoch != 0:
        checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(args.start_epoch - 1))
        model.load_state_dict(checkpoint['state_dict'])





    time_start = time.time()

    print(80 * '=')

    object_dataset = get_test_dataloader(args.imdbval_name, args.num_valid_triplets, args.batch_size, args.num_works)
    test(model, object_dataset, args.num_classes, args.num_test_triplets)

    time_end =time.time()
    print('\n time cost ====== {} s \n'.format(time_end - time_start))

    print(50 * '=')


if __name__ == '__main__':

    mymain()
    # args = parse_args()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # if args.dataset == "pascal_voc":
    #     args.imdb_name = "voc_2007_trainval"
    #     args.imdbval_name = "voc_2007_test"
    # elif args.dataset == "coco":
    #     args.imdb_name = "coco_2017_train"
    #     args.imdbval_name = "coco_2014_minival"
    #
    # l2_dist = PairwiseDistance(2)
    # model = ObjectTripletModel(embedding_size=args.embedding_size, num_classes=args.num_classes).to(device)
    # if args.start_epoch != 0:
    #     checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(args.start_epoch - 1))
    #     model.load_state_dict(checkpoint['state_dict'])


    # path_anc = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
    # path_pos = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
    # path_neg = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000007.jpg'
    #
    # anc_box = [262, 210, 323, 338]
    # pos_box = [4, 243,  66, 373]
    # neg_box = [140,  49, 499, 329]


    # path_anc = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000019.jpg'
    # path_pos = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000028.jpg'
    # path_neg = '/home/zyf/Object_Triplet/data/VOCdevkit2007/VOC2007/JPEGImages/000018.jpg'
    #
    # anc_box = [230, 87, 482, 255]
    # pos_box = [62, 17, 373, 499]
    # neg_box = [30, 29, 357, 278]

    #
    # path_anc = '/home/zyf/AllData/pacal_voc/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2012_004329.jpg'
    # path_pos = '/home/zyf/AllData/pacal_voc/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2012_004331.jpg'
    # path_pos2 = '/home/zyf/AllData/pacal_voc/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg'
    # path_pos3 = '/home/zyf/AllData/pacal_voc/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_002403.jpg'
    # path_neg = '/home/zyf/AllData/pacal_voc/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2010_005230.jpg'
    #
    # anc_box = [56, 87, 284, 396]
    # pos_box = [101, 24, 207, 229]
    # pos2_box = [173, 100, 348, 350]
    # pos3_box = [315, 113, 333, 159]
    # neg_box = [0, 102, 351, 277]
    #
    # img_path = [path_anc, path_pos2, path_neg]
    # boxes = [anc_box, pos2_box, neg_box]
    #
    # pos_dist, neg_dist = getDist(img_path, boxes, model)
    #
    # print(80 * '=')
    # print('anc_pos_dist')
    # print(pos_dist)
    # print('\nanc_neg_dist')
    # print(neg_dist)
    #
    # print('\npos_dist - neg_dist = ' + str(neg_dist.detach().cpu() - pos_dist.detach().cpu()))




