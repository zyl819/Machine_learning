## 数据集
- 1.xlsx PR_feature
- 2.xlsx author_feature
- 3.xlsx PR_info
## 处理
- 三个数据集基于number合并，删去缺省的行
## 运行
你可以这样运行不同模型：
python main.py --model wide_deep
python main.py --model shared_bottom
python main.py --model mmoe
一次性运行所有模型
python main.py --model all
