import os
import torch
from tqdm import tqdm
from utils.utils import get_lr
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
def fit_one_epoch(model_train, model, detr_model,detr_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir,  Freeze_Epoch):
    acc = 0
    loss = 0
    val_loss = 0
    precision = 0
    recall =0
    f1 = 0
    mae_value = 0
    acc_31 = acc_32 = acc_33 = acc_34 = acc_35 =acc_36 = acc_37 = 0
    # print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]   #返回图像和标签
        with torch.no_grad():
            if cuda:
                images = images.cuda(0)
                targets = [{key: ann[key].cuda(0) for key in ann} for ann in targets]

        optimizer.zero_grad()#   清零梯度

        classifier_outputs, detr_loss_value,classifier_losses = model_train(images,targets) #   前向传播
        #判断是否进入分类训练
        if epoch> Freeze_Epoch:
            #loss_value = detr_loss_value + classifier_losses
            loss_value =  classifier_losses
        else:
            loss_value= detr_loss_value

        loss_value.backward()  #   反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) #防止梯度爆炸
        optimizer.step() #更新参数
        loss += loss_value.item()
        pbar.set_postfix(**{'loss': loss / (iteration + 1),'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    # print('Finish Train')

    # print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()#验证验证
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            images = images.cuda(0)
            targets = [{key: ann[key].cuda(0) for key in ann} for ann in targets]
            optimizer.zero_grad()#   清零梯度
            classifier_outputs, detr_loss_value,classifier_losses = model_train(images,targets)  #endtoend模型返回三个值 分类输出、detr损失、分类损失
        if(epoch<= Freeze_Epoch):
            val_loss += detr_loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
        if(epoch> Freeze_Epoch):
            for i in range(7):  #只计算七号牙齿的预测
                predict_dengji = torch.max(classifier_outputs[i], dim=1)[1]   #
                dengji_targets = []
                for j in range(16):  #循环bach  把第i牙位的16个提取出来
                    dengji = int(targets[j]["dengji"][i])
                    dengji_targets.append(dengji)
                tongji =torch.eq(predict_dengji, torch.tensor(dengji_targets).to('cuda:0')).sum().item()  #acc将第相同牙位的bachsize16个预测进行判断正确数

                precision_ = precision_score(dengji_targets, predict_dengji.cpu(),zero_division=0,average='weighted')  # 根据任务选择合适的average方法
                recall_ = recall_score(dengji_targets, predict_dengji.cpu(), zero_division=0,average='weighted')
                f1_ = f1_score(dengji_targets, predict_dengji.cpu(), zero_division=0,average='weighted')
                predict_dengji0 = predict_dengji.to(torch.float32)
                dengji_targets0 = torch.tensor(dengji_targets, dtype=torch.float32).to('cuda:0')
                mae_value_ = torch.mean(torch.abs(predict_dengji0 - dengji_targets0))
                mae_value_ = mae_value_.item()
                # with open('logs/data.txt', 'a') as f:
                #     # 追加数据，确保每次都换行
                #     f.write(f"预测：{predict_dengji}-真实：{dengji_targets}-{tongji}")
                #     if i ==6:
                #         f.write("\n")
                if i ==0:
                    acc_31 = acc_31+tongji
                if i ==1:
                    acc_32 = acc_32+tongji
                if i ==2:
                    acc_33 = acc_33+tongji
                if i ==3:
                    acc_34 = acc_34+tongji
                if i ==4:
                    acc_35 = acc_35+tongji
                if i ==5:
                    acc_36 = acc_36+tongji
                if i ==6:
                    acc_37 = acc_37+tongji

                acc = acc+tongji
                precision = precision+ precision_
                recall = recall+ recall_
                f1 = f1 + f1_
                mae_value =mae_value+mae_value_
            pbar.set_postfix(**{'val_dengji': acc / (7*16)/(iteration + 1),"precision":precision/7/(iteration + 1),"recall":recall/7/(iteration + 1),"f1":f1/7/(iteration + 1),"mae_value":mae_value/7/(iteration + 1)})  # 进度条库的一个方法，允许在进度条旁边附加一些自定义的显示信息
            pbar.update(1)  # 用于将进度条更新一个单位（这里是增加1）
    with open('logs/acc.txt', 'a') as f:
        # 追加数据，确保每次都换行
        f.write(f"{acc_31}-{acc_32}-{acc_33}-{acc_34}-{acc_35}-{acc_36}-{acc_37}")
        f.write("\n")
    acc = acc/(7*16)         #调试使用取消了*100
    precision = precision/7
    recall =recall/7
    f1 = f1/7
    mae_value= mae_value/7
    pbar.close()
    # print('Finish Validation')
    if(epoch<=Freeze_Epoch):
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    else:
        loss_history.append_loss(epoch + 1, loss / epoch_step, acc / epoch_step_val, precision/epoch_step_val,recall/epoch_step_val,f1/epoch_step_val,mae_value/epoch_step_val )
    eval_callback.on_epoch_end(epoch + 1, detr_model)   ###
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ||Val dengji:%.3f' % (loss / epoch_step, val_loss / epoch_step_val,acc/epoch_step_val))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
    #     torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
    #     epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    #if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
    if  (acc / epoch_step_val) >= max(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    #判断是否进入分类阶段，进入后修改优化器
