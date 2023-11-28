import copy
import numpy as np
import random

import cv2

import albumentations as album
import torch
from torch.utils.data import Dataset

from utils.WBAugmenter import WBEmulator as wbAugPython


class ImageMixer(Dataset):
    def __init__(self, root_dir, d_type, fg_datasets, bg_dataset, transform=None, resize_size=(320, 240), fg_to_bg_ratio_range=(0.1, 0.7)): 
        self.d_type = d_type
        self.transform = transform
        self.fg_datasets = fg_datasets
        self.bg_dataset = bg_dataset
        self.fg_to_bg_ratio_range = fg_to_bg_ratio_range
        self.album_transforms = album.Compose(#[album.RandomSizedCrop(min_max_height=[3*resize_size[1], 4*resize_size[1]],
                                              #                       height=resize_size[1], width=resize_size[0],
                                              #                       w2h_ratio=resize_size[1]/resize_size[0])],
                                              [album.Resize(height=resize_size[1], width=resize_size[0]),
                                               album.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.75),
                                               album.ISONoise(p=0.9),
                                               album.AdvancedBlur(p=0.9)],
                                               bbox_params=album.BboxParams(format='pascal_voc',  label_fields=['class_labels']))
        self.p_wb_augment = 0.75
        self.wbAugmenter = wbAugPython.WBEmulator()

    def __resize_fg_img(self, image_fg, image_bg, fg_boxes=None):
        # resize fg image to a random size preserving the aspect ratio
        fg_to_bg_ratio = random.uniform(self.fg_to_bg_ratio_range[0], self.fg_to_bg_ratio_range[1])
        exp_size = (int(fg_to_bg_ratio * image_bg.shape[0]), int(fg_to_bg_ratio * image_bg.shape[1]))

        fx = exp_size[0] / image_fg.shape[0] 
        fy = exp_size[1] / image_fg.shape[1]
        f = min(fx, fy)

        image_fg = cv2.resize(image_fg, (0, 0), fx=f, fy=f)
        if fg_boxes is not None:
            for idx, box in enumerate(fg_boxes):
                fg_boxes[idx] = (box[0]*f, box[1]*f, box[2]*f, box[3]*f)

        return image_fg, fg_boxes

    def __clamp_normalized_box_coords(self, box):
        return np.clip(box, 0., 1.)

    def __check_if_boxes_intersect(self, start_x, start_y, box, box_list):
        box_p = (box[0]+start_x, box[1]+start_y, box[2]+start_x, box[3]+start_y)
        for box_l in box_list:
            if (box_l[0] <= box_p[0] <= box_l[2]) or (box_l[0] <= box_p[2] <= box_l[2]):
                return True
            elif (box_l[1] <= box_p[1] <= box_l[3]) or (box_l[1] <= box_p[3] <= box_l[3]):
                return True

        return False

    def __len__(self):
        return len(self.fg_datasets[0])

    def __getitem__(self, index):
        bg_index = random.randint(0, len(self.bg_dataset)-1)
        image_bg, _ = self.bg_dataset[bg_index]
        image = image_bg
        box_list = []
        label_list = []
        max_num_retries = 50

        for fg_idx, fg_dset in enumerate(self.fg_datasets):
            if fg_idx == 0:
                idx = index
            else:
                idx = random.randint(0, len(fg_dset)-1)
            
            image_fg, (boxes, labels) = fg_dset[idx]
            #resize fg image randomly
            image_fg, boxes = self.__resize_fg_img(image_fg, image_bg, boxes)

            retry = True
            num_retry = 0
            while retry:
                #randomly locate fg on bg
                start_y = random.randint(0, image_bg.shape[0] - image_fg.shape[0] - 1)
                start_x = random.randint(0, image_bg.shape[1] - image_fg.shape[1] - 1)
    
                for box in boxes:
                    retry = False
                    if self.__check_if_boxes_intersect(start_x, start_y, box, box_list):
                        if num_retry < max_num_retries:
                            num_retry += 1
                            retry = True
                            break

                if not retry and (num_retry < max_num_retries):
                    #Fill zero values foreground pizels with background
                    zero_pixels = np.all(image_fg == 0, axis=2)
                    image_fg[zero_pixels] = image_bg[start_y:(start_y + image_fg.shape[0]), start_x:(start_x + image_fg.shape[1]), :][zero_pixels]
                    image[start_y:(start_y + image_fg.shape[0]), start_x:(start_x + image_fg.shape[1]), :] = image_fg
                    for idx, box in enumerate(boxes):
                        box_list.append((box[0]+start_x, box[1]+start_y, box[2]+start_x, box[3]+start_y))
                        label_list.append(labels[idx])

        transform_out = self.album_transforms(image=image, bboxes=box_list, class_labels=label_list)
        image = transform_out['image']
        if np.random.random() < self.p_wb_augment:
            image, wb_pf = self.wbAugmenter.generateWbsRGB(image, 1)
            image = np.array(image[0])
        boxes = transform_out['bboxes']
        labels = transform_out['class_labels']

        if self.transform is not None:
            image = self.transform(image)
            # Normalize boxes:
            for box_idx, box in enumerate(boxes):
                boxes[box_idx] = self.__clamp_normalized_box_coords([float(box[0]/image.shape[2]),
                                                                     float(box[1]/image.shape[1]),
                                                                     float(box[2]/image.shape[2]),
                                                                     float(box[3]/image.shape[1])])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, (boxes, labels)
