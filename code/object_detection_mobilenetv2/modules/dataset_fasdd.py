import os
from pathlib import Path
import numpy as np
import math
import random
import torch
from torch.utils.data import Dataset
import cv2


class FASDDDataset(Dataset):
    '''
    Creates a Pytorch Dataset to train the Object Detector: BED or YoloV1.
    Encodes labels to match the format [xcell, ycell, w, h, confidence, class_0 (smoke), class_1 (fire)]
        - Final encoding format is: [xcell, ycell, w, h, conf=1, smoke?, fire?]

    Discard images when there are more than 1 object in the same cell
    
    In labels .txt
    Fire = 0
    Smoke = 1
    
    Arguments:
        - img_h:            image height
        - img_w:            image width
        - imgs_dir:         path to images folder
        - labels_dir:       path to labels folder
        - file_labels_list: file with all .jpg labels files for train, val or test
        - S:                number of cells in X and Y axis
        - C:                number of classes, 2 in this case
        - max_obj:          maximum number of objects in the picture -> maximum detections
        - ds_len:           lenght of the dataset, to make it shorter
        - transform:        transformation applied to input images -> Albumentations
        - target_transform: transformation applied to labels -> nothing by default

    Return:
        - img:              1 image of the dataset
        - target:           corresponding label encoded
    '''

    def __init__(self, img_h, img_w, imgs_dir, labels_dir, file_labels_list, 
                 S, C, 
                 max_obj,
                 ds_len,
                 ds_rs=False,
                 transform=None, target_transform=None):
        self.img_h = img_h
        self.img_w = img_w
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.file_labels_list = file_labels_list
        self.S = S
        self.C = C
        self.max_obj = max_obj
        self.ds_len = ds_len
        self.ds_rs = ds_rs
        self.transform = transform
        self.target_transform = target_transform

        self.labels_list = self.__get_labels_list__()

        if self.ds_len is not None:
            random.shuffle(self.labels_list)
            self.images_path, self.bboxes, self.labels = self.__build_ds__(self.labels_list[:self.ds_len])
            self.num_samples = self.images_path.shape[0]
        else:
            self.images_path, self.bboxes, self.labels = self.__build_ds__(self.labels_list)
            self.num_samples = self.images_path.shape[0]

            
    def __len__(self):
        return self.num_samples
    
    
    def __get_labels_list__(self):
        labels_list = []
        with open(self.file_labels_list) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                filename = Path(line).stem
                labels_list.append(self.labels_dir + filename + '.txt')
        return labels_list
        
    
    def __bbox_check__(self, bbox):
        eps = 1e-6
        
        xc, yc, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin = xc - w/2
        ymin = yc - h/2
        xmax = xc + w/2
        ymax = yc + h/2
        
        xmin = max(xmin, 0 + eps)
        ymin = max(ymin, 0 + eps)
        xmax = min(xmax, 1 - eps)
        ymax = min(ymax, 1 - eps)
        
        bbox = np.array([ 
                (xmin+xmax)/2,
                (ymin+ymax)/2,
                xmax-xmin,
                ymax-ymin
                 ]).astype(np.float32)
        
        return bbox        


    def __build_ds__(self, labels_list):
        bboxes = []
        labels = []
        images = []
        wrong_imgs = 0
        overlapping_rem = 0
        more_than_x = 0
                
        for label in labels_list:
            fname = Path(label).stem
            if self.ds_rs == True:
                image_path = self.imgs_dir + fname + '.tif'
            else:
                image_path = self.imgs_dir + fname + '.jpg'
            #print(fname, image_path)
                                   
            if cv2.imread(image_path) is None:
                print(f'{image_path} cannot be read by cv2 -> removed')
                wrong_imgs += 1
            
            else:
                
                label_mtx = np.zeros((self.S, self.S))
                overlapping_object = 0

                one_img_bboxes = []
                one_img_labels = []
            
                with open(label) as f:
                    lines = f.readlines()

                    # Restrict to max_obj boxes per sample
                    if len(lines) > self.max_obj:
                        more_than_x += 1
                        continue
                        
                    for line in lines:
                        class_id, x, y, w, h = line.strip().split()
                        class_id = int(class_id)
                        box = np.array([x, y, w, h]).astype(np.float32)
                        x, y, w, h = box[0], box[1], box[2], box[3]
                        box_ok = self.__bbox_check__([x, y, w, h])
                        x, y, w, h = box_ok[0], box_ok[1], box_ok[2], box_ok[3]
                        i, j = math.floor(y * self.S), math.floor(x * self.S)
                        if i > 6:
                            print(f'Box {box} of {fname} has i={i} out of index')
                            print(f'Box checked: {box_ok}')
                            i = 6
                        if j > 6:
                            print(f'Box {box} of {fname} has j={j} out of index')
                            print(f'Box checked: {box_ok}')
                            j = 6
                        if label_mtx[i, j] == 1:
                            overlapping_object = 1
                            overlapping_rem += 1
                            #print(f'Removed {label} due to overlapping object in cell {i, j}')
                            break
                        else:
                            label_mtx[i, j] = 1
                            one_img_bboxes.append([x, y, w, h])
                            # fire: it is the opposite of DFire
                            if class_id == 0:
                                one_img_labels.append(1)
                            # smoke
                            elif class_id == 1:
                                one_img_labels.append(0)
                            else:
                                print(f'File {label} errored in cell {i, j}')

                    if overlapping_object == 0:
                        # Padding to max_obj labels and bounding boxes, so you can store tensors with fixed shape
                        # Label -1 indicates no box
                        for idx in range(self.max_obj - len(one_img_labels)):
                            one_img_bboxes.append([-1, -1, -1, -1])
                            one_img_labels.append(-1)
                        # print(f'\nBboxes and Labels of image {image_path}')
                        # print("Bboxes")
                        # for box in one_bboxes:
                        #     print(box)
                        # print("Labels")
                        # for label in one_labels:
                        #     print(label)
                        bboxes.append(one_img_bboxes)
                        labels.append(one_img_labels)
                        images.append(image_path)
        
        print(f'FASDD Removed wrong images: {wrong_imgs}')
        print(f'FASDD Removed due to overlapping: {overlapping_rem}')
        print(f'FASDD Removed due to more than {self.max_obj}: {more_than_x}')

        labels_np = np.array(labels)
        labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
        bboxes_np = np.array(bboxes)
        bboxes_tensor = torch.tensor(bboxes_np, dtype=torch.float32)
        images_array = np.array(images)
        # print(f'Images array {images_array}')
        # print(f'Bboxes tensor {bboxes_tensor}')
        # print(f'Labels tensor {labels_tensor}')
        
        return images_array, bboxes_tensor, labels_tensor
        #return images, bboxes, labels

    def __getitem__(self, index):

        # Image processing
        img_file = self.images_path[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

        # Labels processing
        bboxes = self.bboxes[index]
        bboxes = bboxes[~torch.all(bboxes == torch.tensor([-1,-1,-1,-1]), dim=1)]
        bboxes = bboxes.numpy().tolist()
        #print(bboxes)
        labels = self.labels[index]
        labels = labels[labels != -1.]
        labels = labels.numpy().tolist()
        #print(f'Labels inside dataset {labels}')
        
        # Data Augmentation
        if self.transform is not None:
            try:
                aug = self.transform(image=img, bboxes=bboxes, class_labels=labels)
                img = aug['image'] / 256.
                bboxes = aug['bboxes']
                labels = aug['class_labels']
            except:
                #print(f'Error trying to augment image {img_file}')
                img = cv2.resize(img, (self.img_w, self.img_h), interpolation = cv2.INTER_NEAREST)
                img = (img / 256.)
                img = torch.tensor(img, dtype=torch.float32)
                img = img.permute(2, 0, 1)
        
        label_mtx = np.zeros((self.S, self.S, 5+self.C))
        
        for box, label in zip(bboxes, labels):
            class_id = int(label)
            i, j = int(box[1]*self.S), int(box[0]*self.S)
            xcell, ycell = box[0]*self.S - j, box[1]*self.S - i
            label_mtx[i, j, :5] = [xcell, ycell, box[2], box[3], 1]
            label_mtx[i, j, 5+class_id] = 1

        label_mtx = torch.tensor(label_mtx, dtype=torch.float32)
        
        #return img, label_mtx, img_file
        return img, label_mtx