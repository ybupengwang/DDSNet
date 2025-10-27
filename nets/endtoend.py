import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .detr_training import HungarianMatcher
from .ops import box_cxcywh_to_xyxy
from PIL import Image
import os
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# class ArcFaceLoss(nn.Module):
#     def __init__(self, num_classes, embedding_size, s=30, m=0.5):
#         super(ArcFaceLoss, self).__init__()
#         self.num_classes = num_classes
#         self.embedding_size = embedding_size
#         self.s = s  # Scaling factor
#         self.m = m  # Margin for the angle
#         self.weights = nn.Parameter(torch.randn(num_classes, embedding_size))  # (num_classes, embedding_size)
#         nn.init.xavier_uniform_(self.weights)
#         self.weights = F.normalize(self.weights, p=2, dim=1)  # Normalize by rows
#     def forward(self, embeddings, labels):
#         normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # (batch_size, embedding_size)
#         cos_theta = F.linear(normalized_embeddings, self.weights)  # (batch_size, num_classes)
#         cos_theta = cos_theta - self.m
#         cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
#         loss = F.cross_entropy(self.s * cos_theta, labels)
#         return loss

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.cross_entropy = torch.nn.CrossEntropyLoss()
#
#     def forward(self, inputs, targets):
#         ce_loss = self.cross_entropy(inputs, targets)
#         pt = torch.exp(-ce_loss)  # 预测概率
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         print("focal_loss和ce_loss:",focal_loss,ce_loss)
#         return 1*focal_loss+1*ce_loss         #用于调整损失选择





class EndToEndDETR(nn.Module):
    def __init__(self, detr_model, small_classifiers,region_model,detr_loss,num_classes, target_size=(224, 224)):
        super(EndToEndDETR, self).__init__()
        # DETR模型作为backbone
        self.detr = detr_model
        # 七个小分类器
        self.small_classifiers = small_classifiers
        self.region_moder = region_model

        self.detr_loss = detr_loss
        self.num_classes = num_classes
        self.target_size = target_size
        self.matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        self.image_index = 0
        # self.cls_num_list = {0:[0,0,12,46,54,43,61,56],1:[0,0,6,40,101,33,63,29],2:[0,0,42,95,93,23,15,4],3:[1,6,43,102,87,17,12,4],
        #                      4:[9,18,68,81,65,18,10,3],5:[0,0,0,11,108,41,80,32],6:[23,33,62,92,52,4,4,2]}
        #添加arcloss
        # self.weightss = nn.Parameter(torch.randn(8, 512)).to('cuda:0')  # 权重矩阵
        #
        # nn.init.xavier_uniform_(self.weightss)
        # self.weightss = F.normalize(self.weightss, p=2, dim=1)  # 归一化
        #
        # self.m = 0.5
        # self.s = 30
        # arcloss


    def forward(self, images, targets=None):


        # 使用DETR模型提取图像特征
        outputs,tezheng = self.detr(images)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        for i, (tensor1, tensor2) in enumerate(indices): #判断一下形状，防止不是7个框
            assert tensor2.size(0) == 7, f"第 {i + 1} 个元组中的第二个张量的长度不是 7，而是 {tensor2.size(0)}"
        idx = self._get_src_permutation_idx(indices)
        # 将预测结果中有对应真实框的预测框取出
        yuce_boxes = outputs['pred_boxes'][idx]
        yuce_boxes = yuce_boxes.reshape(-1, 7, 4)   #变换一下形状，后期待优化

        loss_value = self.detr_loss(outputs, targets)  #
        # print(loss_value)
        weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 1,'cardinality_error':0}
        detr_loss_value = sum(loss_value[k] * weight_dict[k] for k in loss_value.keys() if k in weight_dict)
        #print(detr_loss_value)
        # 使用DETR的预测框（bounding boxes）从分类器中获取裁剪图像
        classified_cropped_images = self.extract_and_classify_cropped_images(images, yuce_boxes, indices)

        #提取ROI区域
        cx = yuce_boxes[:, :, 0]  # 取出每个框的 cx
        cy = yuce_boxes[:, :, 1]  # 取出每个框的 cy
        w = yuce_boxes[:, :, 2]  # 取出每个框的 w
        h = yuce_boxes[:, :, 3]  # 取出每个框的 h
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        x_min_coverage = torch.min(x_min, dim=1).values
        y_min_coverage = torch.min(y_min, dim=1).values
        x_max_coverage = torch.max(x_max, dim=1).values
        y_max_coverage = torch.max(y_max, dim=1).values
        cx_coverage = (x_min_coverage + x_max_coverage) / 2
        cy_coverage = (y_min_coverage + y_max_coverage) / 2
        w_coverage = x_max_coverage - x_min_coverage
        h_coverage = y_max_coverage - y_min_coverage
        region_box = torch.stack([cx_coverage, cy_coverage, w_coverage, h_coverage], dim=1)
        tezheng = []
        for i in range(region_box.shape[0]):
            resized_image = self.crop_and_resize_image(images[i], region_box[i], (512,512))
            tezheng.append(resized_image)

        # region_box = torch.unsqueeze(region_box,1)
        # tezheng = self.extract_and_classify_cropped_images(images, region_box, indices)
        # tezheng = self.crop_and_resize_image(images, region_box, self.target_size)
        tezheng = torch.stack(tezheng)
        tezheng = self.region_moder(tezheng)




        a,_,_,_,_,_,_ = self.small_classifiers(classified_cropped_images[1], tezheng,task_index=0)
        _,b,_,_,_,_,_ = self.small_classifiers(classified_cropped_images[2], tezheng,task_index=0)
        _,_,c,_,_,_,_ = self.small_classifiers(classified_cropped_images[3], tezheng,task_index=0)
        _,_,_,d,_,_,_ = self.small_classifiers(classified_cropped_images[4], tezheng,task_index=0)
        _,_,_,_,e,_,_ = self.small_classifiers(classified_cropped_images[5], tezheng,task_index=0)
        _,_,_,_,_,f,_ = self.small_classifiers(classified_cropped_images[6], tezheng,task_index=0)
        _,_,_,_,_,_,g = self.small_classifiers(classified_cropped_images[7], tezheng,task_index=0)
        classifier_outputs = [a,b,c,d,e,f,g]

        ####  在此处获取batch
        gt = []
        for i in range(16):         #######循环次数需要进行优化  这里用的是bachsize的值
            label = targets[i]['dengji']
            label = label.cpu().numpy()
            gt.append(label)
        transposed_gt = list(zip(*gt))

        classifier_losses = self.compute_classifier_loss(classifier_outputs, transposed_gt)
        return classifier_outputs, detr_loss_value,classifier_losses

    def extract_and_classify_cropped_images(self, images, bbox_tensor, indices):
        batch_size = images.size(0)
        num_boxes = bbox_tensor.size(1)
        # 创建一个字典，存储每个框编号对应的裁剪图像列表
        classified_cropped_images = {i: [] for i in range(1, num_boxes + 1)}
        for i, (_, m) in enumerate(indices):
            image = images[i]  # 获取每张图像
            bboxes = bbox_tensor[i][torch.argsort(m)]  # 获取每张图像的多个框并根据indices顺序进行排序

            # 为每张图像的每个框裁剪并调整大小
            for j in range(num_boxes): ######只获取七号牙位 把num_boxes 换成了（6，7）
                bbox = bboxes[j]  # 获取第 j 个框的坐标 (x_min, y_min, x_max, y_max)
                resized_image = self.crop_and_resize_image(image, bbox, self.target_size)
                classified_cropped_images[j + 1].append(resized_image)  # # 将裁剪后的图像按框编号存储   框编号从 1 开始
        # 将每个框编号的裁剪图像列表转换为 Tensor
        for key in classified_cropped_images:
            classified_cropped_images[key] = torch.stack(classified_cropped_images[key])  # shape: (batch_size, 3, 224, 224)

        return classified_cropped_images

    def crop_and_resize_image(self, image, bbox,target_size):
        # bbox: (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = box_cxcywh_to_xyxy(bbox)*image.size()[1] #将框尺寸恢复到图像大小比例
        if(x_min<0):
           x_min =0
        if(y_min<0):
            y_min =0
        # 测试保存图片数量
        if self.image_index>1903:
            self.image_index = 0
        # 裁剪图片
        cropped_image = image[:, int(y_min):int(y_max), int(x_min):int(x_max)]
        #直接变形
        C, cropped_height, cropped_width = cropped_image.shape
        # 如果裁剪图像的尺寸大于目标尺寸，进行等比例缩放
        if cropped_height > target_size[1] or cropped_width > target_size[0]:
            cropped_resize = transforms.Resize(target_size)
            cropped_image = cropped_resize(cropped_image)
            outtensor= cropped_image

        else:
            cropped_resize = transforms.Resize(target_size)
            cropped_image = cropped_resize(cropped_image)
            outtensor = cropped_image
        #直接变形存放结束
        ##等比例缩放到224*224
        # outtensor = torch.zeros(3, 224, 224).cuda()
        # image_pil = transforms.ToPILImage()(cropped_image)
        # width, height = image_pil.size
        # if width > height:
        #     scale_factor = 224 / width
        # else:
        #     scale_factor = 224 / height
        # # 计算缩放后的新尺寸
        # new_width = int(width * scale_factor)
        # new_height = int(height * scale_factor)
        # resized_pil_image = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # resized_tensor = transforms.ToTensor()(resized_pil_image)
        # if width > height:
        #     h = int((224 - new_height) / 2)
        #     outtensor[:, h:new_height+h, :new_width] = resized_tensor
        #
        # else:
        #     w = int((224 - new_width) / 2)
        #     outtensor[:, :new_height, w:new_width+w] = resized_tensor
        ##等比例缩放结束



        #以下程序是保存剪切后的图象128*128，为了观看detr检测是否正常

        # cropped_image1 = outtensor
        # # 获取缩放后的图像的尺寸
        # _, resized_height, resized_width = cropped_image1.shape
        # cropped_image1 = cropped_image1.permute(1, 2, 0).cpu().numpy()
        # # cropped_image1 = cropped_image1 * np.array([0.229, 0.224, 0.225])
        # # cropped_image1 = cropped_image1 + np.array([0.485, 0.456, 0.406])
        # cropped_image1 = cropped_image1 * 255
        # if cropped_image1.dtype != np.uint8:
        #     cropped_image1 = cropped_image1.astype(np.uint8)
        # cropped_image1 = Image.fromarray(cropped_image1)
        # cropped_image1.save(os.path.join("img_crop128128", "crop_" + str(self.image_index) + ".png"))
        # self.image_index = self.image_index+1
        #保存剪切的图象结束

        #测试一下保存cropped_image 为返回值，进行测试
        #outtensor = ToTensor()(cropped_image1).to('cuda:0')

        return outtensor

    def compute_classifier_loss(self, classifier_outputs, targets):
        losses = []
        total_loss = 0
        for i in range(7):   ##测试只计算第七号牙位把7改成（6,7）
            target_classes = targets[i]  # 获取目标的类标签
            target_classes = torch.tensor(target_classes).cuda()
            output = classifier_outputs[i]
            ##添加arcloss
            # normalized_embeddings = F.normalize(output, p=2, dim=1)
            # cos_theta = F.linear(normalized_embeddings, self.weightss)  # (batch_size, num_classes)
            # #cos_theta = normalized_embeddings @ self.weightss
            # cos_theta = cos_theta - self.m
            # cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
            # arc_loss = F.cross_entropy(self.s * cos_theta, target_classes)

            ##结束arcloss




            # logitadjust开始
            # cls_num_list = torch.cuda.FloatTensor(self.cls_num_list[i])
            # cls_p_list = cls_num_list / 272
            # m_list = 1 * torch.log(cls_p_list)
            # m_list = m_list.view(1, -1)
            # output = output+m_list
            # locitadjust结束

            # 计算分类的交叉熵损失
            loss_ce = F.cross_entropy(output, target_classes)
            #focal loss 开始
            pt = torch.exp(-loss_ce)
            focal_loss = 0.55 * (1 - pt) ** 2 * loss_ce

            #focal loss结束


            # #添加距离损失
            # distance_weights = torch.abs(torch.arange(8, device=output.device).unsqueeze(0) - target_classes.unsqueeze(1))
            # weight_factor = 1.5  # 1.5倍惩罚因子，当距离为1时
            # distance_weights = torch.where(distance_weights == 1, weight_factor * distance_weights, distance_weights)
            # ordinal_loss = loss_ce * distance_weights # 使用目标类别的距离权重
            # loss_ce = ordinal_loss.mean()
            # #距离损失结束

            #只添加相邻损失
            # distance_weights = torch.abs(torch.arange(8, device=output.device).unsqueeze(0) - target_classes.unsqueeze(1))
            # weight_factor = 2  # 1.5倍惩罚因子，当距离为1时
            # distance_weights = torch.where(distance_weights == 1, weight_factor * distance_weights, torch.zeros_like(distance_weights))
            # ordinal_loss = loss_ce * distance_weights # 使用目标类别的距离权重
            # loss_ce = loss_ce + ordinal_loss.mean()
            #只添加相邻损失结束

            loss_ce = 0 * focal_loss + 1 * loss_ce  # focal loss 测试
            losses.append(loss_ce)
        return torch.mean(torch.stack(losses, dim=0))
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx





def resnetload(x,y):
    print('加载Load weights {}.'.format(y))
    model_dict = x.state_dict()
    pretrained_dict = torch.load(y, map_location='cpu')
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    x.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))




