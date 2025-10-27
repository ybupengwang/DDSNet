#
# from functools import partial
# def smallget_lr_scheduler( Init_lr, min_lr, total_iters, step_num=10):
#
#
#     def step_lr(Init_lr, decay_rate, step_size, iters):
#         if step_size < 1:
#             raise ValueError("step_size must above 1.")
#         n = iters // step_size          #0 1 2 3
#         out_lr = Init_lr * decay_rate ** n
#         return out_lr
#
#
#     decay_rate = (min_lr / Init_lr) ** (1 / (step_num - 1))
#     step_size = total_iters / step_num      #100
#     func = partial(step_lr, Init_lr, decay_rate, step_size)
#
#     return func
#
# Init_lr = 0.001
# min_lr = 0.0001
# total_iters = 200
# step_num = 10
#
# # 获取学习率调度器
# lr_scheduler = smallget_lr_scheduler(Init_lr, min_lr, total_iters, step_num)
#
# # 在训练中使用学习率调度器
# for iters in range(total_iters):
#     lr = lr_scheduler(iters)  # 获取当前迭代的学习率
#     print(f"Iteration {iters}, Learning Rate: {lr}")

# from PIL import Image, ImageEnhance
#
# # 打开图像
# image = Image.open('img/image_2.jpg')
#
# # 创建对比度增强器
# enhancer = ImageEnhance.Contrast(image)
#
# # 调整对比度，增强对比度
# image_enhanced = enhancer.enhance(1.8)  # 增强对比度
#
# # 或者减弱对比度
# #image_enhanced = enhancer.enhance(0.5)  # 降低对比度
#
# # 显示增强后的图像
# image_enhanced.show()


#
# import os
#
# def write_folder_names_to_txt(directory, output_file):
#     # 打开输出文本文件，准备写入
#     with open(output_file, 'w') as f:
#         # 遍历指定目录
#         for foldername in os.listdir(directory):
#             # 获取完整的路径
#             folderpath = os.path.join(directory, foldername)
#             # 如果是文件夹
#             if os.path.isdir(folderpath):
#                 f.write(foldername + '\n')  # 写入文件夹名，每个文件夹名字占一行
#
# # 使用示例
# directory = R'D:\dataset\yachi\tianjinkouqing\第二批\21'  # 替换为你的目标文件夹路径
# output_file = R'D:\dataset\yachi\tianjinkouqing\第二批\21\t.txt'      # 输出的txt文件
# write_folder_names_to_txt(directory, output_file)


# import os
# import re
# from collections import defaultdict
#
# # 根目录路径（替换为你的路径）
# root_dir = r'C:\Users\A\Documents\WeChat Files\wxid_01whpwimwdq221\FileStorage\File\2024-10\单个牙齿等级和框选区域核对'
#
# # 存储数据：{名称: [(牙位, 等级)]}
# data_dict = defaultdict(list)
#
# # 遍历所有子目录
# for dirpath, _, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
#             # 匹配文件名，例如：31-B-0001.png
#             match = re.match(r'(\d+)-([A-H])-([^.]+)\.(jpg|png)', filename)
#             if match:
#                 tooth, level, name, _ = match.groups()
#                 data_dict[name].append((int(tooth), level))  # 牙位转为int方便排序
#
# # 写入txt
# output_path = r'C:\Users\A\Documents\WeChat Files\wxid_01whpwimwdq221\FileStorage\File\2024-10\单个牙齿等级和框选区域核对\output.txt'
# with open(output_path, 'w') as f:
#     for name in sorted(data_dict.keys()):
#         sorted_items = sorted(data_dict[name], key=lambda x: x[0])  # 牙位升序
#         levels = [item[1] for item in sorted_items]
#         line = f"{name}," + ",".join(levels)
#         f.write(line + "\n")
#
# print(f"处理完成，结果保存在 {output_path}")

# import pandas as pd
# from sklearn.metrics import cohen_kappa_score, confusion_matrix
# import numpy as np
# # 读取两个专家的CSV文件
# df1 = pd.read_csv('D:\deeplearning\DETER\VOCdevkit\VOC2007\计算一下一致性kappa\专家12一致性.csv')
# df2 = pd.read_csv('D:\deeplearning\DETER\VOCdevkit\VOC2007\计算一下一致性kappa\专家3一致性.csv')
#
# # 检查 image_name 是否对齐
# if not df1['name'].equals(df2['name']):
#     raise ValueError("两个CSV文件的图像顺序不一致")
#
# cols = df1.columns[1:8]
#
# kappa_list = []
# po_list = []
# pe_list = []
#
# print("牙位\tKappa\t类别数\tPo(观察一致性)\tPe(偶然一致性)")
# print("-" * 55)
#
# for col in cols:
#     y1 = df1[col]
#     y2 = df2[col]
#
#     # 清除缺失值
#     valid_idx = ~(y1.isna() | y2.isna())
#     y1 = y1[valid_idx]
#     y2 = y2[valid_idx]
#
#     # 计算 Cohen's Kappa
#     kappa = cohen_kappa_score(y1, y2)
#
#     # 混淆矩阵
#     labels = sorted(set(y1.unique()) | set(y2.unique()))
#     cm = confusion_matrix(y1, y2, labels=labels)
#
#     total = cm.sum()
#     Po = np.trace(cm) / total
#     row_marginals = cm.sum(axis=1)
#     col_marginals = cm.sum(axis=0)
#     Pe = np.sum(row_marginals * col_marginals) / (total ** 2)
#
#     # 保存每列结果
#     kappa_list.append(kappa)
#     po_list.append(Po)
#     pe_list.append(Pe)
#
#     print(f"{col}\t{kappa:.3f}\t{len(labels)}\t\t{Po:.3f}\t\t{Pe:.3f}")
#
# # 输出平均结果
# mean_kappa = np.mean(kappa_list)
# mean_po = np.mean(po_list)
# mean_pe = np.mean(pe_list)
#
# print("-" * 55)
# print(f"平均\t{mean_kappa:.3f}\t\t\t{mean_po:.3f}\t\t{mean_pe:.3f}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 真实标签 & 预测标签
y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
y_pred = [0, 0, 1, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]

# 各类别评估（不平均）
print("各类别精确率:", precision_score(y_true, y_pred, average=None))
print("各类别召回率:", recall_score(y_true, y_pred, average=None))
print("各类别F1分数:", f1_score(y_true, y_pred, average=None))

# 宏观/微观/加权平均
print("\nMacro平均精确率:", precision_score(y_true, y_pred, average='macro'))
print("Macro平均召回率:", recall_score(y_true, y_pred, average='macro'))
print("Macro平均F1分数:", f1_score(y_true, y_pred, average='macro'))

print("\nMicro平均精确率:", precision_score(y_true, y_pred, average='micro'))
print("Micro平均召回率:", recall_score(y_true, y_pred, average='micro'))
print("Micro平均F1分数:", f1_score(y_true, y_pred, average='micro'))

print("\nWeighted平均精确率:", precision_score(y_true, y_pred, average='weighted'))
print("Weighted平均召回率:", recall_score(y_true, y_pred, average='weighted'))
print("Weighted平均F1分数:", f1_score(y_true, y_pred, average='weighted'))

# 综合指标
print('\n准确率:', accuracy_score(y_true, y_pred))

# 分类报告（包含所有关键指标）
print("\n分类报告:\n", classification_report(y_true, y_pred))
