from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.roi_data_layer.roidb import combine_roidb

def getAllRoI(imdb_name):
    imdb, roidb, ratio_list, ratio_index = combine_roidb(imdb_name)

    ##################################################################################
    ##################################################################################
    # dataset pascal_voc
    AllRoI = dict()

    if(imdb_name.split('_')[0] == "coco"):
        class_21 = [str(i) for i in list(range(81))]
        class_index = list(range(81))
    else:

        class_21 = ['__background__',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']
        class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for i in class_index:
        AllRoI[class_21[i]] = []

    for img_index, img_roidb in enumerate(roidb):  #
        if len(img_roidb['gt_classes']) >= 1:
            for box_index, perbox in enumerate(img_roidb['boxes']):
                AllRoI[class_21[img_roidb['gt_classes'][box_index]]].append([img_index, box_index])

    ##################################################################################
    ##################################################################################

    return AllRoI, class_21, roidb
