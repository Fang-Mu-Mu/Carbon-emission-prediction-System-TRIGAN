# -*- coding:utf-8 -*-
import copy
import os
import sys
from Mmodels1 import SAEG_XLTNet
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from Mget_data1 import setup_seed

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

setup_seed(123)
class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    def forward(self, y, y_pred):
        residual = y-y_pred
        loss = torch.max((self.quantile - 1) * residual, self.quantile * residual)
        return torch.mean(loss)
q=0.95
@torch.no_grad()
def get_val_loss(args, model, Val, edge_index):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    # loss_function = QuantileLoss(q)
    val_loss = []
    for (seq, labels) in Val:
        seq = seq.to(args.device)
        labels = labels.to(args.device)  # (batch_size, n_outputs, pred_step_size)
        preds = model(seq, edge_index)
        total_loss = 0
        for k in range(args.input_size):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)



def plot(y, pred, ind, label):
    # plot
    plt.plot(y[:500], color='blue', label='true value')

    plt.plot(pred[:500], color='red', label='pred value')
    plt.title('第' + str(ind) + '变量预测示意图')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def get_sde(y,pred):
    return np.std(y-pred)

def get_IC(y,pred):
    n = len(y)
    numerator = np.sqrt(np.sum((pred - y)**2) / n)
    denominator = np.sqrt(np.sum(pred**2) / n) + np.sqrt(np.sum(y**2) / n)
    IC = numerator / denominator
    return IC
import numpy as np


def get_nse(y,pred):
    # 计算观测值的平均值
    mean_obs = np.mean(y)
    # 计算分子：观测值与模拟值之差的平方和
    sum_sq_error = np.sum((y - pred) ** 2)
    # 计算分母：观测值与观测值平均值之差的平方和
    sum_sq_tot = np.sum((y - mean_obs) ** 2)
    # 计算NSE
    if sum_sq_tot == 0:
        return float('inf')  # 如果观测值没有变化，则返回无穷大
    else:
        return 1 - (sum_sq_error / sum_sq_tot)


def get_wi(y, pred):
    # 计算观测值的平均值
    mean_obs = np.mean(7)
    # 计算分子：预测值与观测值之差的平方和
    sum_sq_diff = np.sum((pred - y) ** 2)
    # 计算分母：预测值与观测值平均值之差的绝对值之和的平方
    sum_sq_total = np.sum((np.abs(pred - mean_obs) + np.abs(y - mean_obs)) ** 2)
    # 计算WI
    WI = 1 - (sum_sq_diff / sum_sq_total)
    return WI


def get_LMI(y, pred):

    # 计算观测值的平均值
    mean_obs = np.mean(y)

    # 计算分子：预测值与观测值之差的绝对值之和
    sum_abs_diff = np.sum(np.abs(pred - y))

    # 计算分母：观测值与观测值平均值之差的绝对值之和
    sum_abs_dev = np.sum(np.abs(y - mean_obs))

    # 计算LMI
    if sum_abs_dev == 0:
        return float('inf')  # 如果观测值没有变化，则返回无穷大
    else:
        return 1 - (sum_abs_diff / sum_abs_dev)

def get_nmse(y, pred):
    # 计算真实值的平均值
    mean_true = np.mean(y)
    # 计算分子：预测值与真实值之差的平方和
    sum_sq_error = np.sum((y - pred) ** 2)
    # 计算分母：真实值与真实值平均值之差的平方和
    sum_sq_total = np.sum((y - mean_true) ** 2)
    # 计算NMSE
    if sum_sq_total == 0:
        return float('inf')  # 如果真实值没有变化，则返回无穷大
    else:
        return sum_sq_error / sum_sq_total


def get_mspe(y, pred):
    epsilon = np.finfo(float).eps
    y_true = np.array(y)
    y_pred = np.array(pred)
    mask = y_true != 0
    # 计算MSPE
    mspe_value = np.mean(np.square((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
    return mspe_value

def practice(args, Pract, edge_index):
    edge_index = edge_index.to(args.device)
    model = SAEG_XLTNet(args)
    # loss_function = nn.MSELoss().to(args.device)
    loss_function = QuantileLoss(q)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, labels) in Pract:
            seq = seq.to(args.device)
            labels = labels.to(args.device)  # (batch_size, n_outputs, pred_output_size)

            # print(labels.shape)
            # print(preds.shape)
            total_loss = 0
            # for k in range(args.input_size):
            #     total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
            #
            # total_loss = total_loss / preds.shape[0]
            # total_loss.requires_grad_(True)
            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            # train_loss.append(total_loss.item())
            #train_loss.requires_grad_(True)
            optimizer.zero_grad()
            preds = model(seq, edge_index)  # (n_outputs, batch_size, pred_output_size)将结果堆叠为张量
            loss = loss_function(preds[0, :, :], labels[:, 0, :])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Pract, edge_index)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            state = {'model': best_model.state_dict()}
            torch.save(state, r'D:\Users\三木\Desktop\论文/stgcn.pkl')

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
    import os
    state = {'model': best_model.state_dict()}
    torch.save(state, r'D:\Users\三木\Desktop\论文/stgcn.pkl')


def practice_test(args, Pract, scaler, edge_index):
    print('loading models...')
    edge_index = edge_index.to(args.device)
    model = SAEG_XLTNet(args)
    model.load_state_dict(torch.load(r'D:\Users\三木\Desktop\论文/stgcn.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for (seq, targets) in tqdm( Pract):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(args.input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        seq = seq.to(args.device)
        with torch.no_grad():
            _pred = model(seq, edge_index)
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    # for (seq, targets) in tqdm( Pract):
    #     targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
    #     target = targets[:, 0, :]
    #     target = list(chain.from_iterable(target))
    #     ys.append(target)
    #     seq = seq.to(args.device)
    #     with torch.no_grad():
    #         _pred = model(seq, edge_index)
    #         pred = _pred[0,:,:]
    #         pred = list(chain.from_iterable(pred.data.tolist()))
    #         preds.append(pred)

    # ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    # pred = np.array(preds).reshape(-1, 1)
    # y = (np.array(ys)).reshape(-1, 1)
    y=ys[0,:]
    pred=preds[0,:]
    # 测试集数据反归一化
    # pred = scaler.inverse_transform(predictions)
    # y = scaler.inverse_transform(labels)
    # y = scaler.inverse_transform(ys).T
    # pred = scaler.inverse_transform(preds).T
    mses, rmses, maes, mapes,sdes,nse,r2,wi,nmse,lmi,mspe,tic = [], [], [], [],[],[],[],[],[],[],[],[]
    print('第', str(1), '个序列:')
    print('nmse', get_nmse(y, pred))
    print('rmse:', get_rmse(y, pred))
    print("MSPE", get_mspe(y, pred))
    print('mape:', get_mape(y, pred))
    print('NSE', get_nse(y, pred))
    print('SDE:', get_sde(y, pred))
    print("TIC", get_IC(y, pred))
    print("LMI", get_LMI(y, pred))
    print('wi', get_wi(y, pred))
    print('mae:', get_mae(y, pred))
    print('r2',get_r2(y,pred))
    print('mse:', get_mse(y, pred))



    nmse.append(get_nmse(y, pred))
    mses.append(get_mse(y, pred))
    rmses.append(get_rmse(y, pred))
    maes.append(get_mae(y, pred))
    mapes.append(get_mape(y, pred))
    sdes.append(get_mape(y,pred))
    nse.append(get_nse(y,pred))
    r2.append(get_r2(y,pred))
    wi.append(get_wi(y,pred))
    tic.append(get_IC(y,pred))
    lmi.append(get_LMI(y,pred))
    mspe.append(get_mspe(y, pred))
    print('--------------------------------')
    plt.plot(y[:500], color='blue', label='true value')
    plt.plot(pred[:500], color='red', label='pred value')
    plt.title('上海碳排放预测示意图')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()
    df = {"nmse":nmse,"rmse": rmses,"mspe":mspe,"NSE":nse,"SDE":sdes,"wi":wi,"lmi":lmi,"mse": mses,
          "mae": maes, "mape": mapes,"r2":r2}
    df = pd.DataFrame(df)
    f = pd.DataFrame({'Actual': y[:500],'Predicted':pred[:500]})
    df.to_csv(r'D:\Users\三木\Desktop\论文/result.csv')
    f.to_csv(r'D:\Users\三木\Desktop/prediction.csv')


def predict_future_30days(args, model_path, scaler, edge_index, test_loader, future_days=30):
    """
    预测未来30天未知数据
    基于测试集最后的数据序列进行预测
    """
    print('Loading model for future prediction...')

    # 加载模型
    edge_index = edge_index.to(args.device)
    model = SAEG_XLTNet(args)  # 根据您的实际模型类名调整
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()

    print('Extracting last sequence from test data...')

    # 从测试集中获取最后一个序列作为预测的起点
    last_sequence = None
    last_targets = None

    # 遍历测试集获取最后一个batch的最后一个序列
    for (seq, targets) in test_loader:
        last_sequence = seq
        last_targets = targets

    if last_sequence is None:
        raise ValueError("No data found in test_loader")

    # 取最后一个batch的最后一个样本作为预测起点
    # 假设last_sequence的形状为 [batch_size, seq_len, features]
    start_sequence = last_sequence[-1:, :, :]  # 取最后一个样本
    start_sequence = start_sequence.to(args.device)

    print(f'Predicting next {future_days} days...')

    all_predictions = []
    current_sequence = start_sequence

    # 递归预测未来多天
    for day in tqdm(range(future_days)):
        with torch.no_grad():
            # 使用当前序列进行预测
            pred = model(current_sequence, edge_index)

            # 根据您的模型输出结构调整这里的索引
            # 假设pred的形状为 [batch_size, pred_len, features]
            daily_pred = pred[:, -1:, :]  # 取最后一个预测时间步

            # 保存预测结果
            daily_pred_np = daily_pred.cpu().numpy()
            all_predictions.append(daily_pred_np)

            # 更新序列：移除最旧的数据，添加新的预测数据
            current_sequence = update_prediction_sequence(current_sequence, daily_pred, args)

    # 处理预测结果
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)

        # 重塑为2D数组用于反归一化
        if all_predictions.ndim == 3:
            all_predictions_2d = all_predictions.reshape(-1, all_predictions.shape[-1])
        else:
            all_predictions_2d = all_predictions.reshape(-1, 1)

        # 反归一化预测结果
        future_predictions = scaler.inverse_transform(all_predictions_2d)

        # 如果预测的是多个特征，取第一个特征（碳排放）
        if future_predictions.ndim == 2 and future_predictions.shape[1] > 1:
            future_predictions = future_predictions[:, 0]

        return future_predictions.flatten()
    else:
        return np.array([])


def update_prediction_sequence(current_sequence, new_pred, args):
    """
    更新输入序列：移除最旧的数据，添加新的预测数据
    根据您的模型输入结构进行调整
    """
    # 假设current_sequence的形状为 [batch_size, seq_len, features]
    # 移除第一个时间步
    updated_sequence = current_sequence[:, 1:, :]

    # 添加新的预测作为最后一个时间步
    # 确保new_pred的形状与序列一致 [batch_size, 1, features]
    updated_sequence = torch.cat([updated_sequence, new_pred], dim=1)

    return updated_sequence


def plot_future_predictions(predictions, last_known_date=None, future_days=30):
    """
    绘制未来预测结果
    """
    # 生成未来日期
    if last_known_date is not None:
        start_date = pd.Timestamp(last_known_date) + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=future_days, freq='D')
    else:
        # 如果没有提供最后日期，使用默认日期
        future_dates = pd.date_range(start="2025-01-31", periods=future_days, freq='D')

    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, predictions, color='red', linewidth=2, label='Future Prediction')
    plt.scatter(future_dates, predictions, color='darkred', s=30, zorder=5)

    plt.title('Future 30 Days Carbon Emission Prediction')
    plt.xlabel('Date')
    plt.ylabel('Carbon Emission (Mt)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 美化图表
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return future_dates


def save_future_predictions(predictions, future_dates, filename=None):
    """
    保存预测结果到CSV文件
    """
    if filename is None:
        filename = r'D:\Users\三木\Desktop\论文/future_30days_prediction.csv'

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Carbon_Emission': predictions
    })

    future_df.to_csv(filename, index=False)
    print(f'Future predictions saved to {filename}')

    return future_df
