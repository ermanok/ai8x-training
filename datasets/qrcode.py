import numpy as np
import random
from urllib.request import Request, urlopen

import cv2
import qrcode
import albumentations as album
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.bg20k import BG20K
from datasets.image_mixer_v2 import ImageMixer

import ai8x
from utils import object_detection_utils


class RandomWordGenerator():
    def __init__(self):
        url = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        web_byte = urlopen(req).read()
        long_txt = web_byte.decode('utf-8')
        self.words = long_txt.splitlines()

    def get_random_word(self):
        idx = random.randint(0, len(self.words)-1)
        return self.words[idx]


class QRCodeGenerator(Dataset):
    min_qr_size = 2
    max_qr_size = 10
    min_num_words = 1
    max_num_words = 4

    def __init__(self, root_dir, d_type, data_len, transform=None, augment_data=False):
        self.data_len = data_len
        self.transform = transform

        self.random_text_list = []
        self.augment_data = augment_data

        self.__gen_random_words()
        
        if self.augment_data:
            self.geometric_transforms = album.Compose([album.Affine(scale = (0.6, 1.3),
                                                                translate_percent=(0.2, 0.4),
                                                                rotate=(-45, 45),
                                                                shear=(-45, 45),
                                                                mode=cv2.BORDER_CONSTANT,
                                                                fit_output=True,
                                                                p=0.9),],
                                                       #album.Perspective(scale=(0.05, 0.3), p=0.5),],
                                                       bbox_params=album.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            self.chromatic_transforms = album.Compose([album.RGBShift(r_shift_limit=64,
                                                                      g_shift_limit=64,
                                                                      b_shift_limit=64, 
                                                                      p=0.9),
                                                       album.ColorJitter(brightness=0.5,
                                                                         contrast=0.5,
                                                                         saturation=0.5,
                                                                         hue=0.5, p=0.9),
                                                       album.MultiplicativeNoise(multiplier=(0.5, 1.5),
                                                                                 per_channel=True,
                                                                                 elementwise=True,
                                                                                 p=0.7),
                                                       album.MotionBlur(p=0.7),],
                                                       bbox_params=album.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __gen_random_words(self):
        random_word_gen = RandomWordGenerator()#RandomWords()

        for _ in range(self.data_len):
            num_words = random.randint(self.min_num_words, self.max_num_words)
            
            text = random_word_gen.get_random_word()
            for _ in range(num_words-1):
                text = ' '.join([text, random_word_gen.get_random_word()])
            self.random_text_list.append(text)
    
    def __gen_qr(self, text):
        qr_size = random.randint(self.min_qr_size, self.max_qr_size)

        qr = qrcode.QRCode(version=1,
                           error_correction=qrcode.constants.ERROR_CORRECT_L,
                           box_size=qr_size,
                           border=2)

        qr.add_data(text)
        qr.make(fit=True)

        return qr.make_image(fill_color="black", back_color="white")

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        text = self.random_text_list[index]
        qr_image = self.__gen_qr(text)
        qr_image = 250 * np.asarray(qr_image).astype(np.uint8) + 5# convert to numpy

        image = qr_image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        start_x = start_y = 2
        boxes = [[start_x, start_y, start_x + qr_image.shape[1]-1-2, start_y+qr_image.shape[0]-1-2]]
        labels = [1]
        
        if self.augment_data:
            transform_out = self.geometric_transforms(image=image, bboxes=boxes, class_labels=labels)
            image = transform_out['image']
            zero_pixels = np.all(image == 0, axis=2)
            #print(image.shape, zero_pixels.sum())
            transform_out = self.chromatic_transforms(image=image, bboxes=transform_out['bboxes'], class_labels=transform_out['class_labels'])
            image = transform_out['image']
            image[zero_pixels] = 0
            boxes = transform_out['bboxes']
            labels = transform_out['class_labels']

        if self.transform is not None:
            image = self.transfom(image)

        return image, (boxes, labels)


def QRCode_get_datasets(data, load_train=True, load_test=True, im_size=(320, 240), fg_to_bg_ratio_range=(0.1, 0.7), num_qr_per_img=1):
    (data_dir, args) = data

    train_dataset = test_dataset = None
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         ai8x.normalize(args=args)])

    if load_train:
        bg_dataset = BG20K(root_dir=data_dir, d_type='train', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train', data_len=10000, augment_data=True))

        train_dataset = ImageMixer(data_dir, 'train', fg_dataset, bg_dataset, transform=data_transform, resize_size=im_size, fg_to_bg_ratio_range=fg_to_bg_ratio_range)
    
    if load_test:
        bg_dataset = BG20K(root_dir=data_dir, d_type='test', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train', data_len=2000, augment_data=True))

        test_dataset = ImageMixer(data_dir, 'test', fg_dataset, bg_dataset, transform=data_transform, resize_size=im_size, fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    return train_dataset, test_dataset


def QRCode_qVGA_get_datasets(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.05, 0.95))


def QRCode_qVGA_get_datasets_numqrs2(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.15, 0.45), num_qr_per_img=2)


def QRCode_qVGA_get_datasets_numqrs3(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.15, 0.35), num_qr_per_img=3)


def QRCode_qVGA_get_datasets_qrsize_10(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.05, 0.15))


def QRCode_qVGA_get_datasets_qrsize_20(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.15, 0.25))


def QRCode_qVGA_get_datasets_qrsize_30(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.25, 0.35))


def QRCode_qVGA_get_datasets_qrsize_40(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.35, 0.45))


def QRCode_qVGA_get_datasets_qrsize_50(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.45, 0.55))


def QRCode_qVGA_get_datasets_qrsize_60(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.55, 0.65))


def QRCode_qVGA_get_datasets_qrsize_70(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.65, 0.75))


def QRCode_qVGA_get_datasets_qrsize_90(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qVGA (320x240) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(320, 240), fg_to_bg_ratio_range=(0.80, 0.95))


def QRCode_qqVGA_get_datasets(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qqVGA (160x120) resolution"""
    return QRCode_get_datasets(data, load_train, load_test, im_size=(160, 120), fg_to_bg_ratio_range=(0.05, 0.95))


datasets = [
    {
        'name': 'QRCode',
        'input': (3, 240, 320),
        'output': ([1]),
        'loader': QRCode_qVGA_get_datasets,
        'collate': object_detection_utils.collate_fn
    },
    {
        'name': 'QRCode_160_120',
        'input': (3, 120, 160),
        'output': ([1]),
        'loader': QRCode_qqVGA_get_datasets,
        'collate': object_detection_utils.collate_fn
    }
]

