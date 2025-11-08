from Margs1 import args_parser
from Mget_data1 import nn_seq, setup_seed
from Mmodels1 import SAEG_XLTNet
from Muilts1 import practice, practice_test
from Mget_data import adj2coo
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
setup_seed(42)
def main():
    args = args_parser()
    train_loader,val_loader,test_loader, scaler, edge_index = nn_seq(args)
    practice(args, train_loader, edge_index)
    practice_test(args, test_loader,scaler, edge_index)
# def main():
#     args = args_parser()
#     train_loader, val_loader, test_loader, scaler, edge_index = nn_seq(args)
#
#     # 1. 训练模型
#     practice(args, train_loader, edge_index)
#
#     # 2. 测试模型
#     practice_test(args, test_loader, scaler, edge_index)

    # # 3. 预测未来30天 - 直接在这里写代码
    # print("\n" + "=" * 60)
    # print("开始预测未来30天碳排放数据")
    # print("=" * 60)
    #
    # # 加载训练好的模型
    # print('加载模型用于未来预测...')
    # edge_index = edge_index.to(args.device)
    # model = SAEG_XLTNet(args)
    # model.load_state_dict(torch.load(r'D:\Users\三木\Desktop\论文/stgcn.pkl')['model'])
    # model.eval()
    #
    # # 从测试集中获取最后一个序列作为预测起点
    # print('从测试数据中提取最后一个序列...')
    # last_sequence = None
    # for (seq, targets) in test_loader:
    #     last_sequence = seq
    #     break
    #
    # if last_sequence is None:
    #     print("错误：测试加载器中未找到数据")
    #     return
    #
    # # 准备初始序列
    # if len(last_sequence.shape) == 3:
    #     start_sequence = last_sequence[-1:, :, :]  # 取最后一个样本
    # else:
    #     start_sequence = last_sequence.unsqueeze(0)
    #
    # start_sequence = start_sequence.to(args.device)
    # print(f"初始序列形状: {start_sequence.shape}")
    #
    # # 开始预测未来30天
    # print('预测未来30天...')
    # all_predictions = []
    # current_sequence = start_sequence.clone()
    #
    # for day in tqdm(range(30)):
    #     with torch.no_grad():
    #         # 使用当前序列进行预测
    #         pred = model(current_sequence, edge_index)
    #
    #         # 取最后14个时间步，然后取其中的最后一个
    #         last_14_preds = pred[:, -14:, :]  # [1, 14, features]
    #         daily_pred = last_14_preds[:, -1:, :]  # 取14个中的最后一个 [1, 1, features]
    #
    #         # 保存预测结果
    #         daily_pred_np = daily_pred.cpu().numpy()
    #         all_predictions.append(daily_pred_np)
    #
    #         # 更新序列：移除第一个时间步，添加新的预测
    #         updated_sequence = current_sequence[:, 1:, :]
    #         current_sequence = torch.cat([updated_sequence, daily_pred], dim=1)
    #
    # # 处理预测结果
    # all_predictions = np.concatenate(all_predictions, axis=0)
    # print(f"预测结果形状: {all_predictions.shape}")
    #
    # # 重塑为2D数组用于反归一化
    # if all_predictions.ndim == 3:
    #     all_predictions_2d = all_predictions.reshape(-1, all_predictions.shape[-1])
    # else:
    #     all_predictions_2d = all_predictions.reshape(-1, 1)
    #
    # # 反归一化
    # future_predictions = scaler.inverse_transform(all_predictions_2d)
    #
    # # 如果预测的是多个特征，取第一个特征（碳排放）
    # if future_predictions.ndim == 2 and future_predictions.shape[1] > 1:
    #     future_predictions = future_predictions[:, 0]
    #
    # future_predictions = future_predictions.flatten()
    #
    # # 生成未来日期
    # future_dates = pd.date_range(start="2025-01-1", periods=30, freq='D')
    #
    # # 绘制预测结果
    # plt.figure(figsize=(12, 6))
    # plt.plot(future_dates, future_predictions, color='red', linewidth=2, label='未来预测')
    # plt.scatter(future_dates, future_predictions, color='darkred', s=30, zorder=5)
    # plt.title('未来30天碳排放预测')
    # plt.xlabel('日期')
    # plt.ylabel('碳排放 (Mt)')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    #
    # # 保存预测结果
    # future_df = pd.DataFrame({
    #     'Date': future_dates,
    #     'CO2 (Mt)': future_predictions
    # })
    #
    # future_df.to_csv(r'D:\Users\三木\Desktop\修改\预测数据\Inner Mongolia.csv', index=False)
    #
    # # 打印预测统计信息
    # print(f'\n未来30天预测统计信息:')
    # print(f'平均排放量: {np.mean(future_predictions):.2f} Mt')
    # print(f'标准差: {np.std(future_predictions):.2f} Mt')
    # print(f'最小值: {np.min(future_predictions):.2f} Mt')
    # print(f'最大值: {np.max(future_predictions):.2f} Mt')
    # print(f'30天总排放量: {np.sum(future_predictions):.2f} Mt')
    #
    # # 显示详细预测值
    # print(f'\n未来30天详细预测值:')
    # for i in range(len(future_predictions)):
    #     print(f'第{i + 1}天 ({future_dates[i].strftime("%Y-%m-%d")}): {future_predictions[i]:.2f} Mt')

# def main_with_future():
#     args = args_parser()
#     train_loader, val_loader, test_loader, scaler, edge_index = nn_seq(args)
#
#     # 训练和测试
#     practice(args, train_loader, edge_index)
#     practice_test(args, test_loader, scaler, edge_index)
#
#     # 额外：预测未来
#     future_predictions = predict_future_30days(
#         args=args,
#         model_path=r'D:\Users\三木\Desktop\论文/stgcn.pkl',
#         scaler=scaler,
#         edge_index=edge_index,
#         test_loader=test_loader,
#         future_days=30
#     )
#
#     return future_predictions
if __name__ == '__main__':
    main()

    # main_with_future()
