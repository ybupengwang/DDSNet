
import datetime
import os
from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import Counter
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.detr import DETR
from nets.detr_training import (build_loss, get_lr_scheduler,set_optimizer_lr,weights_init,dengji_lr_decay)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DetrDataset, detr_dataset_collate
from utils.utils import (get_classes, seed_everything, worker_init_fn,WeightedRandomSampler)
from utils.utils_fit import fit_one_epoch
from nets.endtoend import (EndToEndDETR,resnetload)
from nets.smallmodel import FPNresnet34
from nets.smallmodelresnet import resnet34
from nets.vit import vit_base_patch16_224_in21k
if __name__ == "__main__":
    Cuda            = True
    seed            = 11
    classes_path    = 'model_data/cls_classes.txt'  #牙位分类
    dengji_path = 'model_data/dengji_classes.txt'   #牙齿成熟度分类
    vit_weigth_path = "model_data/vit_base_patch16_224_in21k.pth"
    resnet34_weight_path = "model_data/resnet34-333f7ec4.pth"
    # resnet34_weight_path = ""
    detr_model_path      = 'model_data/detr_resnet50_weights_coco.pth' #加载detr预训练权重
    # detr_model_path =""
    #endtoend_path   = 'logs/loss_2024_11_20_21_24_05/ep300-loss0.613-val_loss0.560.pth' #调试用  原先的预训练权重
    endtoend_path = 'logs/loss_2025_01_14_08_39_22/best_epoch_weights.pth'                       #加载endtoend预训练权重，可以修改Init_Epoch跳过detr训练，直接进行牙齿成熟度分类训练
    #endtoend_path = ''
    input_shape     = [1200, 1200]   #   input_shape     输入的shape大小 [高 宽】
    backbone        = "resnet50" #detr模型利用的是resnet50作为backbone
    #pretrained               与detr_model_path和Freeze_Train 有关。
    #                         不设置detr_model_path，pretrained = True，此时仅加载主干开始训练。
    #                         不设置detr_model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程
    #------------------------------------------------------------------#
    pretrained = False
    Init_Epoch          = 101  #从哪个epoch开始 单独训练牙齿分类的时候使用
    Freeze_Epoch        = 100 #目前 Freeze_Epoch代表前300轮只训练detr，超过300后，冻结detr开始训练7个小模型
    Freeze_batch_size   = 16   #目前没用预留
    UnFreeze_Epoch      = 600  #训练总epoch
    Unfreeze_batch_size = 16   #bachsize

    Freeze_Train        = True #冻结训练
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adamw"
    momentum            = 0.9
    weight_decay        = 1e-4
    lr_decay_type       = "cos" #使用到的学习率下降方式，可选的有step、cos
    save_period         = 10 #多少个epoch保存一次权值

    save_dir            = 'logs'#权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 100
    #------------------------------------------------------------------#
    #   官方提示为TODO this is a hack
    #   稳定性未知，默认为不开启
    #------------------------------------------------------------------#
    aux_loss            = False
    num_workers         = 4
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    seed_everything(seed)
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dengji_names, tooth_class_numbers = get_classes(dengji_path) #获取牙齿分类等级名字和数量
    class_names, num_classes = get_classes(classes_path)#获取牙位名字和数量
    detr_loss = build_loss(num_classes)
    detr_model = DETR(backbone, 'sine', 256, num_classes, 50, pretrained=pretrained)
    #加载detr预训练权重
    if detr_model_path != '':
        resnetload(detr_model,detr_model_path )

    #加载vit模型权重
    small_classifiers = vit_base_patch16_224_in21k(num_classes=8, has_logits=False).to('cuda:0')
    if vit_weigth_path != "":
        resnetload(small_classifiers, vit_weigth_path)
    #加载结束
    #加载区域特征模型权重
    region_model = resnet34(include_top=False)
    if resnet34_weight_path !="":
        resnetload(region_model,resnet34_weight_path)


    #small_classifiers = FPNresnet34()

    # small_classifiers1 = FPNresnet34()
    # small_classifiers2 = FPNresnet34()
    # small_classifiers3 = FPNresnet34()
    # small_classifiers4 = FPNresnet34()
    # small_classifiers5 = FPNresnet34()
    # small_classifiers6 = FPNresnet34()
    # small_classifiers7 = FPNresnet34()


    #加载小分类模型权重
    # if resnet34_weight_path !="":
    #     resnetload(small_classifiers1, resnet34_weight_path)
    #     resnetload(small_classifiers2, resnet34_weight_path)
    #     resnetload(small_classifiers3, resnet34_weight_path)
    #     resnetload(small_classifiers4, resnet34_weight_path)
    #     resnetload(small_classifiers5, resnet34_weight_path)
    #     resnetload(small_classifiers6, resnet34_weight_path)
    #     resnetload(small_classifiers7, resnet34_weight_path)
    #创建端到端模型


    model = EndToEndDETR(detr_model, small_classifiers,region_model,detr_loss, tooth_class_numbers, target_size=(224, 224))  #8:代表牙类别 后期需要进行优化

    #加载endtoend模型权重
    if endtoend_path != '' and Init_Epoch !=0:
        resnetload(model, endtoend_path)
        for param in model.detr.parameters():  #如果加载了endtoend模型，并且只训练分类就把detr冻结
            param.requires_grad = False
    # ------------------------------------#
    #   冻结detr中的backbone权重训练
    # ------------------------------------#
    if Freeze_Train:
        for param in model.detr.backbone.parameters():
            param.requires_grad = False
    #VIT模型预训练权重加载和解冻
    for name, param in model.small_classifiers.named_parameters():
        param.requires_grad = False
        if 'head' in name  or 'norm' in name or 'linear_layer' in name or '11.mlp' in name or 'crossattentionblock' in name :
            param.requires_grad = True
    # for param in model.small_classifiers.blocks[:1]:
    #     for p in param.parameters():
    #         p.requires_grad = True
    #VIT模型预训练权重加载和解冻结束
    #区域特征提取参数冻结
    for name, param in model.region_moder.named_parameters():
        param.requires_grad = False
        if 'bn' in name:
            param.requires_grad = True



    #小分类模型
    # for name, param in model.small_classifiers.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'fpn' in name or 'attention' in name:
    #         param.requires_grad = True



    # for name, param in model.small_classifiers2.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    # for name, param in model.small_classifiers3.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    # for name, param in model.small_classifiers4.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    # for name, param in model.small_classifiers5.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    # for name, param in model.small_classifiers6.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    # for name, param in model.small_classifiers7.named_parameters():
    #     param.requires_grad = False
    #     if 'fc' in name or 'bn' in name or 'attention' in name:
    #         param.requires_grad = True
    #----------------------#
    #   记录Loss
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    model_train     = model.train()
    #cudnn.benchmark = True
    model_train = model_train.cuda()
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)


    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    if optimizer_type in ['adam', 'adamw']:
        Init_lr_fit = Init_lr
        Min_lr_fit  = Min_lr
    #---------------------------------------#
    #   根据optimizer_type选择优化器
    #---------------------------------------#
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Init_lr_fit / 10,
        },
    ]
    optimizer = {
        'adam'  : optim.Adam(param_dicts, Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
        'adamw' : optim.AdamW(param_dicts, Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
        'sgd'   : optim.SGD(param_dicts, Init_lr_fit, momentum = momentum, nesterov=True, weight_decay=weight_decay),
    }[optimizer_type]
    lr_scale_ratio = [1, 0.1]

    #---------------------------------------#
    #   获得学习率下降的公式
    #---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    #---------------------------------------#
    #   判断每一个世代的长度
    #---------------------------------------#
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # all_labels = []
    # for train_line in train_lines:
    #     boxes = train_line.split()[7:]  # 提取所有框的信息
    #     for box in boxes:
    #         label = int(box.split(',')[5])  # 假设标签是第6个字段
    #         all_labels.append(label)

    # 统计每个类别的出现频率
    # label_counts = Counter(all_labels)
    # total_samples = len(all_labels)
    # class_weights = {label: total_samples / count for label, count in label_counts.items()}
    # # 为每个样本分配权重，样本权重越小的类别权重越大
    # sample_weights = []
    # for annotation in train_lines:
    #     boxes = annotation.split()[6:]  # 提取第几号牙齿所有框的信息
    #     weight = 0
    #     for box in boxes:
    #         label = int(box.split(',')[5])  # 标签
    #         weight += class_weights[label]  # 样本的权重是所有目标框类别权重的总和
    #     sample_weights.append(weight)
    sample_weights = None
    #统计结束
    #---------------------------------------#
    #   构建数据集加载器。
    #---------------------------------------#
    train_dataset   = DetrDataset(train_lines, input_shape, num_classes,sample_weights, train = True)
    #添加样本权重
    #sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset), replacement=True)
    val_dataset     = DetrDataset(val_lines, input_shape, num_classes, train = False)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last=False, collate_fn=detr_dataset_collate,worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    gen_val         = DataLoader(val_dataset  , shuffle = False, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last=False, collate_fn=detr_dataset_collate,worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
    #----------------------#
    #   记录eval的map曲线
    #----------------------#
    eval_callback   = EvalCallback(detr_model, input_shape[0], class_names, dengji_names, num_classes,val_lines, log_dir, Cuda,eval_flag=eval_flag, period=eval_period)
    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        batch_size = Unfreeze_batch_size

        if optimizer_type in ['adam', 'adamw']:
            Init_lr_fit = Init_lr
            Min_lr_fit  = Min_lr
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch) #衰减类型  初始值10的-4   最小值10的-6  轮数600
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        if(epoch <= Freeze_Epoch):  #训练detr用余弦退火损失
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch, lr_scale_ratio)
        else:                       #训练分类模型用多项式损失
            epoch_dengji = epoch-Freeze_Epoch
            max_steps = UnFreeze_Epoch-Freeze_Epoch
            #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch_dengji:  dengji_lr_decay(epoch_dengji,max_steps))
            #以下也是多项式学习率
            a = 0.002 * (1 - epoch_dengji / max_steps) ** 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = a
            #scheduler.step()
        if (epoch == Freeze_Epoch + 1):#训练分类模型切换adam优化器
            optimizer = optim.AdamW(model.parameters(), lr=0.01,betas = (0.9, 0.999), weight_decay=1e-4)
            for param in model.detr.parameters():  # 进入分类训练时把detr所有参数冻结
                param.requires_grad = False
            for param in model.detr.backbone.parameters():  # 进入分类训练时把backbone所有参数冻结
                param.requires_grad = False

        fit_one_epoch(model_train, model, detr_model,detr_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, log_dir,  Freeze_Epoch)

    loss_history.writer.close()
#测试查看冻结情况[(name, param.requires_grad) for name, param in model.named_parameters()]