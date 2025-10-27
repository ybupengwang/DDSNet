import os
import cv2
import numpy as np
import torch
from PIL import Image,ImageEnhance
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


def save_images_from_numpy(numpy_array, output_dir='./'):
    # 确保输入是一个 4D 数组，形状为 [batch_size, height, width, channels]
    if len(numpy_array.shape) != 4:
        raise ValueError('Input numpy array must have 4 dimensions (batch_size, height, width, channels)')

    batch_size, height, width, channels = numpy_array.shape

    # 遍历每一张图片
    for i in range(batch_size):
        img = numpy_array[i]  # 获取单张图片 [height, width, channels]

        if channels == 3:
            # 如果是RGB图像，确保通道顺序是 BGR 格式，因为 OpenCV 使用 BGR 格式
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif channels == 1:
            # 如果是灰度图像，直接使用
            img = img.squeeze(axis=-1)  # 去掉最后的单通道维度

        # 保存图片
        cv2.imwrite(f'{output_dir}/image_{i}.png', img)
        print(f'Saved image_{i}.png')

class DetrDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes,sample_weights=None,train= False): #添加了样本权重
        super(DetrDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train
        #样本权重设置
        if sample_weights is None:
            self.sample_weights = np.ones(self.length)
        else:
            self.sample_weights = np.array(sample_weights)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        #print(self.annotation_lines[index])
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1)) #图像转换为 NumPy 数组 对图像进行标准化处理 图像的维度顺序进行转置
        box         = np.array(box, dtype=np.float32)

        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  #计算出框的中心点坐标，预测出来需要再转 需要注意 （cx,cy,w,h）
        else:
            box = np.zeros([0, 5], np.float32)
        # dengji_label = box[:, 5]  # labels 是最后一列
        # dengji_label = np.array(dengji_label,dtype=np.float32)
        # dengji_labels =
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a


    def get_random_data(self, annotation_line, input_shape, jitter=.3, random=False):
        line = annotation_line.split()
        image = Image.open(line[0])
        #增加对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        image = cvtColor(image)

        iw, ih = image.size
        w, h = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        scalew = w / iw
        scaleh = h / ih
        image = image.resize((w, h), Image.BICUBIC)
        # image.save(os.path.join("img_crop", "cropda_" + str(0) + ".png")) 测试用
        flip = self.rand() < .5
        #flip = 0
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)

        box[:, [0, 2]] = box[:, [0, 2]] * scalew
        box[:, [1, 3]] = box[:, [1, 3]] * scaleh
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0  # box[:, 0:2]返回【N,2】 box[:, 0:2] < 0 是条件
        box[:, 2][box[:, 2] > w] = w  # 判断是否超过w
        box[:, 3][box[:, 3] > h] = h  # 判断是否超过h
        box_w = box[:, 2] - box[:, 0]  # box的宽
        box_h = box[:, 3] - box[:, 1]  # box的高
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # 过滤掉宽度或高度小于等于 1 的目标框
        dengji_label = box[:, 5]

        return image_data, box

    
# DataLoader中collate_fn使用
def detr_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # labels = [torch.from_numpy(ann[:, 4]).type(torch.FloatTensor) for ann in bboxes]
    # boxes  = [torch.from_numpy(ann[:, :4]).type(torch.FloatTensor) for ann in bboxes]

    targets = [
        {
            "labels": torch.from_numpy(ann[:, 4]).type(torch.LongTensor), 
            "boxes" : torch.from_numpy(ann[:, :4]).type(torch.FloatTensor),
            "dengji": torch.from_numpy(ann[:, 5]).type(torch.LongTensor),
        } for ann in bboxes
    ]
    return images, targets


def cxcywh_to_xyxy(box):
    # 获取中心点和宽高
    cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    # 计算 xmin, ymin, xmax, ymax
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return np.stack([xmin, ymin, xmax, ymax], axis=-1)
