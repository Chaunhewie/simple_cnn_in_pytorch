# coding=utf-8
import os
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
import sys

All_People_Indexes = ['S119', 'S130', 'S127', 'S073', 'S067', 'S107', 'S092', 'S160', 'S134', 'S106', 'S101', 'S155',
                      'S109', 'S053', 'S116', 'S139', 'S064', 'S117', 'S505', 'S099', 'S122', 'S082', 'S079', 'S121',
                      'S066', 'S115', 'S102', 'S050', 'S131', 'S071', 'S093', 'S077', 'S062', 'S055', 'S011', 'S034',
                      'S506', 'S014', 'S074', 'S087', 'S059', 'S010', 'S029', 'S091', 'S058', 'S113', 'S085', 'S136',
                      'S097', 'S068', 'S061', 'S114', 'S057', 'S045', 'S149', 'S156', 'S054', 'S051', 'S108', 'S110',
                      'S105', 'S135', 'S999', 'S151', 'S504', 'S026', 'S083', 'S502', 'S503', 'S028', 'S100', 'S005',
                      'S158', 'S078', 'S060', 'S080', 'S133', 'S112', 'S501', 'S065', 'S075', 'S069', 'S096', 'S124',
                      'S147', 'S137', 'S063', 'S132', 'S128', 'S022', 'S037', 'S086', 'S138', 'S046', 'S032', 'S120',
                      'S125', 'S094', 'S042', 'S072', 'S070', 'S052', 'S089', 'S076', 'S088', 'S148', 'S090', 'S056',
                      'S118', 'S129', 'S103', 'S095', 'S084', 'S154', 'S157', 'S104', 'S111', 'S035', 'S044', 'S098',
                      'S126', 'S895', 'S081']
# random.shuffle(All_People_Indexes)
# print(All_People_Indexes)


class CKPlus(data.Dataset):
    """`CK+ Dataset & CK+48 Dataset, CK+48 is aborted after adding face location detect and crop operation.
    Args:
        is_train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version.
                                        E.g, ``transforms.RandomCrop``
        target_type(str, optional): Using for target type: "fa" for "float array", "ls" for "long single".
                                    E.g, ``MSELoss will use fa``; ``CrossEntropyLoss will use ls``
        k_folder (int, optional): Using for split the dataset as train set and test set,
                                  and len(test set):len(train set) = 1:(10-k_folder).
        img_dir_pre_path (str, optional): The relative path of the data dictionary and main file.
        using_fl (bool, optional): Whether using face_landmarks to crop original img.

        there are 981(anger:135 contempt:54 disgust:177 fear:75 happy:207 sadness:84 surprise:249) images in data with 123 people
        we choose images of 111 people, whose name is in self.train_people_names, for training
        we choose images of 12 person, whose name is in self.test_people_names, for testing
    """

    def __init__(self, is_train=True, transform=None, k_folder=1, img_dir_pre_path="data/CK+"):
        self.classes_map = {'anger': 0,
                     'contempt': 1,
                    'disgust': 2,
                    'fear': 3,
                    'happy': 4,
                    'sadness': 5,
                    'surprise': 6}
        self.img_dir_pre_path = img_dir_pre_path
        self.transform = transform
        self.is_train = is_train  # train set or test set
        self.name = 'CK+'

        # 分割所有的人物编号，分成测试集和训练集
        split_index = int(len(All_People_Indexes)*k_folder/10)
        if split_index < 1:
            split_index = 1
        self.train_people_indexes = All_People_Indexes[:-split_index]
        self.test_people_indexes = All_People_Indexes[-split_index:]

        self.train_data = []
        self.train_data_num = 0
        self.train_classes = []
        self.test_data = []
        self.test_data_num = 0
        self.test_classes = []
        classes = os.listdir(self.img_dir_pre_path)
        for c in classes:
            img_file_names = os.listdir(os.path.join(self.img_dir_pre_path, c))
            for img_file_name in img_file_names:
                if img_file_name[:4] in self.train_people_indexes:
                    self.train_data_num += 1
                    if is_train:
                        img = Image.open(os.path.join(self.img_dir_pre_path, c, img_file_name)).convert("L")
                        self.train_data.append(img)
                        self.train_classes.append(self.classes_map[c])
                elif img_file_name[:4] in self.test_people_indexes:
                    self.test_data_num += 1
                    if not is_train:
                        img = Image.open(os.path.join(self.img_dir_pre_path, c, img_file_name)).convert("L")
                        self.test_data.append(img)
                        self.test_classes.append(self.classes_map[c])
                else:
                    print("img:(%s,%s) is not belong to both of train or test set!" % (c, img_file_name))
        print("train_num: ", self.train_data_num, " test_num:", self.test_data_num)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index >= self.__len__():
            return None, None

        if self.is_train:
            img, cla = self.train_data[index], self.train_classes[index]
        else:
            img, cla = self.test_data[index], self.test_classes[index]

        # 由于存在 random_crop 等的随机处理，应该是读取的时候进行，这样每个epoch都能够获取不同的random处理
        if self.transform is not None:
            img = self.transform(img)
        return img, cla

    def __len__(self):
        """
        Returns:
            int: data num.
        """
        if self.is_train:
            return self.train_data_num
        else:
            return self.test_data_num
        
    def set_transform(self, transform):
        self.transform = transform
