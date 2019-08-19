from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import os.path as osp
import numpy as np
import argparse
from skimage import io
import tqdm
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from lib.roi_data_layer.roidb import combine_roidb
from lib.model.triplet import ObjectTripletModel
from lib.model.tripletutils import PairwiseDistance, TripletLoss
from lib.model.eval_metrics import evaluate, plot_roc
from lib.model.dataloader import get_dataloader
from lib.roi_data_layer.getAllRoI import getAllRoI

from lib.roi_data_layer.minibatch import get_minibatch
from lib.roi_data_layer.roibatchLoader import roibatchLoader

Loss_All = []
Accurucy_All = []
Threshold = []


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
                        default = 300, type = int)
    parser.add_argument('--save-dir', dest = 'save_dir',
                        default = './models', type = str)
    parser.add_argument('--num-train-triplets', dest = 'num_train_triplets',
                        default = 5000, type = int)
    parser.add_argument('--num-valid-triplets', dest = 'num_valid_triplets',
                        default = 2000, type = int)
    parser.add_argument('--batch-size', dest = 'batch_size',
                        default = 32, type = int)
    parser.add_argument('--num_works', dest = 'num_works',
                        default = 4, type = int)


    args = parser.parse_args()

    return args



def train_valid(All_epoch_train_loss, All_epoch_val_loss, model, optimizer, scheduler, epoch, dataloaders, data_size):
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    l2_dist = PairwiseDistance(2)
    train_loss_iter = []
    val_loss_iter = []

    for phase in ['train', 'valid']:
        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()

        # for batch_idx, batch_sample in tqdm(enumerate(dataloaders[phase])):
        for batch_sample in tqdm(dataloaders[phase]):# 118(iter) * 17(bs) = 2000
            anc_img = batch_sample['anc_img'][0].to(device)
            # print(anc_img.shape)
            pos_img = batch_sample['pos_img'][0].to(device)
            neg_img = batch_sample['neg_img'][0].to(device)

            # pos_cls = batch_sample['pos_class'].to(device)
            # neg_cls = batch_sample['neg_class'].to(device)

            with torch.set_grad_enabled(phase == 'train'):
                # print(80 * '*')
                # print(model)
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the hard negatives only for training
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                # print('pos_dist ===> ')
                # print(torch.mean(pos_dist))
                neg_dist = l2_dist.forward(anc_embed, neg_embed)
                # print('neg_dist ===> ')
                # print(torch.mean(neg_dist))

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                # print('allallallallallallallall')
                # print(all)  # num of batch_size

                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets].to(device)
                pos_hard_embed = pos_embed[hard_triplets].to(device)
                neg_hard_embed = neg_embed[hard_triplets].to(device)

                anc_hard_img = anc_img[hard_triplets].to(device)
                pos_hard_img = pos_img[hard_triplets].to(device)
                neg_hard_img = neg_img[hard_triplets].to(device)

                # pos_hard_cls = pos_cls[hard_triplets].to(device)
                # neg_hard_cls = neg_cls[hard_triplets].to(device)

                anc_img_pred = model.forward_classifier(anc_hard_img).to(device)
                pos_img_pred = model.forward_classifier(pos_hard_img).to(device)
                neg_img_pred = model.forward_classifier(neg_hard_img).to(device)

                # print(args.margin)
                triplet_loss = TripletLoss(args.margin).forward(anc_hard_embed,
                                                                pos_hard_embed, neg_hard_embed).to(device)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                dists = l2_dist.forward(anc_embed, pos_embed)
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # print(dists)
                # print(dists.data.cpu().numpy())
                # print(np.ones(dists.size(0)))
                distances.append(dists.data.cpu().numpy())
                labels.append(np.ones(dists.size(0)))

                dists = l2_dist.forward(anc_embed, neg_embed)
                distances.append(dists.data.cpu().numpy())
                labels.append(np.zeros(dists.size(0)))

                triplet_loss_sum += triplet_loss.item()
                if phase == 'train':
                    train_loss_iter.append(triplet_loss.item())
                    # All_epoch_train_loss.append(triplet_loss.item())
                else:
                    val_loss_iter.append(triplet_loss.item())
                    # All_epoch_val_loss.append(triplet_loss.item())


        avg_triplet_loss = triplet_loss_sum / data_size[phase] * args.batch_size
        if phase == 'train':
            All_epoch_train_loss.append(avg_triplet_loss)
        else:
            All_epoch_val_loss.append(avg_triplet_loss)
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        # print('distance -----  distance')
        # print('len = ' + str(len(distances))) #related to train/valid set
        # print(distances)

        tpr, fpr, accuracy, val, val_std, far, best_threshold = evaluate(distances, labels)

        if not os.path.exists('./log/train_loss/'):
            os.makedirs('./log/train_loss/')
        if not os.path.exists('./log/val_loss/'):
            os.makedirs('./log/val_loss/')
        if not os.path.exists('./log/roc_val/'):
            os.makedirs('./log/roc_val/')
        if not os.path.exists('./log/all_epoch/'):
            os.makedirs('./log/all_epoch/')

        plt.cla()
        if phase == 'train':
            plt.plot(train_loss_iter)
            plt.plot(len(train_loss_iter) * [sum(train_loss_iter)/len(train_loss_iter)])
            plt.savefig('./log/train_loss/train_loss_epoch_{}.png'.format(epoch))
            plt.close()
            plt.plot(All_epoch_train_loss)
            plt.savefig('./log/all_epoch/all_epoch_train_loss_epoch_{}.png'.format(epoch))
            plt.close()
        else:
            plt.plot(val_loss_iter)
            plt.plot(len(val_loss_iter) * [sum(val_loss_iter) / len(val_loss_iter)])
            plt.savefig('./log/val_loss/val_loss_epoch_{}.png'.format(epoch))
            plt.close()
            plt.plot(All_epoch_val_loss)
            plt.savefig('./log/all_epoch/all_epoch_val_loss_epoch_{}.png'.format(epoch))
            plt.close()

        print(' {} set - Triplet Loss    = {:.8f}'.format(phase, avg_triplet_loss))
        print(' {} set - Accuracy        = {:.8f}'.format(phase, np.mean(accuracy)))
        print('\n -----------------------------------------------------------------')
        print(' Best threshold is {} '.format(best_threshold))
        Loss_All.append(avg_triplet_loss)
        Accurucy_All.append(np.mean(accuracy))
        Threshold.append(best_threshold)

        with open('./log/{}_log_epoch.txt'.format(phase), 'w') as f:
            f.write('epoch' + '\t' +
                    'Accuracy' + '\t' +
                    'Loss' + '\t' +
                    'Threshold' + '\n')
            for k in range(len(Loss_All)):
                f.write(str(k//2)     + '\t' +
                        str(Accurucy_All[k]) + '\t' +
                        str(Loss_All[k]) + '\t' +
                        str(Threshold[k]) + '\n')

        if phase == 'train':
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       './log/checkpoint_epoch{}.pth'.format(epoch))
        else:
            plot_roc(fpr, tpr, figure_name = './log/roc_val/roc_valid_epoch_{}.png'.format(epoch))

    return All_epoch_train_loss, All_epoch_val_loss


class sampler(Sampler):
    def __int__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

def mymain():

    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    l2_dist = PairwiseDistance(2)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"


    model = ObjectTripletModel(embedding_size=args.embedding_size, num_classes=args.num_classes).to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    # evry step_size epoch down lr gamma
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

    if args.start_epoch != 0:
        checkpoint = torch.load('./log/checkpoint_epoch{}.pth'.format(args.start_epoch - 1))
        model.load_state_dict(checkpoint['state_dict'])


    All_epoch_train_loss = []
    All_epoch_val_loss = []

    for epoch in range(args.start_epoch, args.num_epochs + args.start_epoch):
        time_start = time.time()

        print(80 * '=')
        print('Epoch [{} / {}]'.format(epoch, args.num_epochs + args.start_epoch - 1))

        data_loaders, data_size = get_dataloader(args.imdb_name, args.imdbval_name, args.num_train_triplets, args.num_valid_triplets, args.batch_size, args.num_works)

        All_epoch_train_loss, All_epoch_val_loss = train_valid(All_epoch_train_loss, All_epoch_val_loss, model, optimizer, scheduler, epoch, data_loaders, data_size)

        time_end =time.time()
        print('\n time cost of  epoch {} ====== {} s \n'.format(epoch, time_end - time_start))

    print(50 * '=')




if __name__ == '__main__':

    # args = parse_args()
    # # print('Called with args: ')
    # # print(args)
    # # print(torch.cuda.is_available())
    #
    # if args.dataset == "pascal_voc":
    #     print('dataset is pascal_voc')
    #     args.imdb_name = "voc_2007_trainval"
    #     args.imdbval_name = "voc_2007_test"
    # elif args.dataset == "coco":
    #     print('dataset is coco')
    #     args.imdb_name = "coco_2017_train"
    #     args.imdbval_name = "coco_2017_val"
    #
    # train_size = len(roidb)
    # print('train_size = ' + str(train_size))
    #
    # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # print('output_dir is ' + output_dir)

    # print(imdb.image_index)
    # print(roidb[0])
    # print(roidb[1])
    # print(len(roidb[1]['gt_classes']))

    # test coco data
    # imdb, roidb, ratio_list, ratio_index = combine_roidb(args.imdbval_name)
    # print(roidb)
    # print('roidb num ')
    # print(len(roidb))

    print('=====> start =====> \n\n')
    mymain()

    # print('#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#')
    # print(imdb.name)
    # print(roidb[2]['boxes'][1])
    # ppp = roidb[2]['boxes'][1]
    # print(roidb[2]['gt_classes'][1])
    # print(roidb[2]['image'])
    # img = cv2.imread(roidb[2]['image'], cv2.IMREAD_COLOR)
    # print(img)
    # print(img.shape)
    # print(20 * '=')
    #
    # img2 = io.imread(roidb[2]['image'])
    # print(img2)
    # print(img2.shape)
    # img = img[ppp[1]:ppp[3], ppp[0]:ppp[2], :]
    # cv2.imwrite('img.png', img)
    # # print(img)
    # print('---------------------------')
    # print(roidb[2]) # flipped = False
    # print('---------------------------')
    # print(roidb[5013]) # flipped = True
    # ###0802 end get voc
    # print('##############main####################')
    # mymain()