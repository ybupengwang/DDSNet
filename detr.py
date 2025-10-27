import colorsys
import os
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from nets.detr import DETR
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox
from nets.endtoend import EndToEndDETR
from nets.detr_training import (build_loss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from nets.smallmodel import FPNresnet34
from nets.vit import vit_base_patch16_224_in21k
from nets.smallmodelresnet import resnet34
'''
训练自己的数据集必看注释！
'''
detr_loss = build_loss(7)
class Detection_Transformers(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : r'logs/loss_2025_01_13_08_31_53/best_epoch_weights.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        "dengji_path"       :'model_data/dengji_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小
        #---------------------------------------------------------------------#
        "min_length"        : 800,
        # "input_shape" : [1200, 1200],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   主干网络的种类
        #---------------------------------------------------------------------#
        "backbone"          : 'resnet50',
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化detr
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.dengji_names,_ =get_classes(self.dengji_path)
        self.bbox_util                      = DecodeBox()

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        small_classifiers = vit_base_patch16_224_in21k(num_classes=8, has_logits=False).to('cuda:0')
        region_model = resnet34(include_top=False)
        small_classifiers2 = FPNresnet34().to('cuda:0')
        small_classifiers3 = FPNresnet34().to('cuda:0')
        small_classifiers4 = FPNresnet34().to('cuda:0')
        small_classifiers5 = FPNresnet34().to('cuda:0')
        small_classifiers6 = FPNresnet34().to('cuda:0')
        small_classifiers7 = FPNresnet34().to('cuda:0')

        #---------------------------------------------------#
        #   建立detr模型，载入detr模型的权重
        #---------------------------------------------------#
        detr= DETR('resnet50', 'sine', 256, self.num_classes, num_queries=50)
        self.net= EndToEndDETR(detr, small_classifiers,region_model,detr_loss, 8, target_size=(224, 224))
        #self.net    = DETR(self.backbone, 'sine', 256, self.num_classes, num_queries=100)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        self.smalloutput1 = self.net.small_classifiers
        # self.smalloutput2 = self.net.small_classifiers2
        # self.smalloutput3 = self.net.small_classifiers3
        # self.smalloutput4 = self.net.small_classifiers4
        # self.smalloutput5 = self.net.small_classifiers5
        # self.smalloutput6 = self.net.small_classifiers6
        # self.smalloutput7 = self.net.small_classifiers7
        self.net = self.net.detr

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, self.min_length)

        #image_data = image.resize((self.input_shape[0],self.input_shape[0]),Image.BICUBIC)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            #self.input_shape = torch.unsqueeze(torch.from_numpy(np.array(self.input_shape)), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()
                #self.input_shape = self.input_shape.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs,tezheng= self.net(images)


            results = self.bbox_util(outputs, images_shape, self.confidence)

            #去重开始
            # tensor = results
            # last_column = tensor[0][:, -1].to(torch.int)
            # second_last_column = tensor[0][:, -3]
            # _, unique_indices = torch.unique(last_column, return_inverse=True)
            # result = []
            # processed_tensors = []
            # for idx in range(last_column.unique().size(0)):
            #     # 获取当前重复值的所有行
            #     current_rows = tensor[0][unique_indices == idx]
            #     current_values = second_last_column[unique_indices == idx]
            #
            #     # 获取倒数第二列最大值对应的行
            #     max_value_index = torch.argmax(current_values)
            #
            #     # 保留最大值的行
            #     result.append(current_rows[max_value_index])
            # result_tensor = torch.stack(result)
            # processed_tensors.append(result_tensor)
            # # #去重结束
            # results = processed_tensors
            if results[0] is None: 
                return image
            
            _results    = results[0].cpu().numpy()
            top_label   = np.array(_results[:, 5], dtype = 'int32')
            top_conf    = _results[:, 4]
            top_boxes   = _results[:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // self.min_length, 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            dengji = {}
            for i, c in list(enumerate(top_label)):

                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                yawei = self.class_names[int(c)]
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image_tensor = ToTensor()(crop_image.resize((224,224))).unsqueeze(0).to('cuda:0')
                m,_,_,_,_,_,_ = self.smalloutput1(crop_image_tensor,tezheng,0)
                if c ==0:

                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                    print(smalloutput)
                if c == 1:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                if c == 2:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                if c ==3:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                if c ==4:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                if c ==5:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                if c ==6:
                    smalloutput = torch.argmax(torch.softmax(m, dim=1))
                    dengji_name = self.dengji_names[int(smalloutput)]
                    dengji[yawei] = (dengji_name,int(smalloutput))
                #crop_image.save(os.path.join(dir_save_path, "crop_" +str(c)+ ".png"), quality=95, subsampling=0)
                #print("save crop_" + str(i) + ".png to " + dir_save_path)
        print(dict(sorted(dengji.items())))
        return dengji
        #######添加了只保留分数最大值的七科牙齿， 目前中切牙不好检测
        # unique_keys = {}
        # result_values = np.full_like(top_conf,fill_value=np.nan,dtype=float)
        # for k, v in zip(top_label, top_conf):
        #     if k not in unique_keys:
        #         unique_keys[k] = v
        #     else:
        #         unique_keys[k] = max(unique_keys[k], v)
        # for i,k in enumerate(top_label):
        #     if k in unique_keys and unique_keys[k] ==top_conf[i]:
        #         result_values[i] = top_conf[i]
        # top_conf = result_values
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):

            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            if np.isnan(score):
                continue

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            x1,y1,x2,y2 = font.getbbox(label)
            label_size = (x2-x1, y2-y1)
            #label_size = draw.textsize(label, font)   #######################修改了textsize 替换成了textlength
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, self.min_length)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()

            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                images          = torch.from_numpy(image_data)
                images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
                if self.cuda:
                    images          = images.cuda()
                    images_shape    = images_shape.cuda()
                
                outputs = self.net(images)
                results = self.bbox_util(outputs, images_shape, self.confidence)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, self.min_length)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)
                                                    
            if results[0] is None: 
                return 
            
            _results    = results[0].cpu().numpy()
            top_label   = np.array(_results[:, 5], dtype = 'int32')
            top_conf    = _results[:, 4]
            top_boxes   = _results[:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
