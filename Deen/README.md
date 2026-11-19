# Apple counting 本地基线

在 VS Code 中打开 E:/MV/apple_counting，并选择 .venv/Scripts/python.exe。

## 运行顺序

所有命令在项目根目录 PowerShell 中运行。

```powershell
.\.venv\Scripts\python.exe check_gpu.py
.\.venv\Scripts\python.exe prepare_detection_data.py --source data/detection/train
.\.venv\Scripts\python.exe train_apples.py --check-data
.\.venv\Scripts\python.exe train_apples.py --epochs 1 --batch 4 --name apple_smoke
.\.venv\Scripts\python.exe predict_apples.py --baseline
```

首次短训练只验证流程，不代表最终性能。随后可以用 --epochs 30 --name apple_baseline 训练基线。

## 数据

官方入口：https://rsn.umn.edu/MinneApple
下载 detection 数据，解压后将 images 和 masks 放到 data/detection/train/ 下。
不能用只有图片、没有实例标注的数据冒充有标签训练集。
脚本将非零实例 ID 转为 YOLO 检测框，按文件名序列前缀分组划分 train/val，输出 manifest.csv、summary.json 和预览图。
分组依赖文件名约定，仍需检查相邻场景和拍摄来源，不能据此宣称完全没有场景泄漏。

## 结果

runs/detect：训练记录、检测指标、best.pt。
runs/counting：每张图片的标注数、预测数、误差、MAE、RMSE、平均偏差，以及标注/预测对照图。
计数对照数据集标注；数量相同不等于定位正确。验证集用于开发，不是独立测试集。不要把通用 COCO 模型的预测直接当作已训练的苹果计数模型结果。

## Colab 迁移

上传代码和数据，安装 requirements.txt。Colab 使用系统已有的兼容 GPU PyTorch，无需复制 Windows .venv。
脚本支持 --data 路径及 --device 0 或 cpu。重新运行训练会生成当前机器的数据绝对路径。
迁移已有权重时显式传入 predict_apples.py --weights /content/.../best.pt --data /content/.../apple_yolo。
课程要求主项目开发及演示在 Colab 完成，须在 Colab 实际运行验证。

## 当前范围

本阶段只搭建苹果计数。完整小组项目还要求电阻色环任务、连贯方法及传统图像处理与机器学习的结合。成员贡献应按实际工作填写，不预填作者。

## Experiment folders

Training names are automatic: model_epN_szN_bN_timestamp. --name adds an optional label.
Counting outputs: runs/counting/<training experiment>/<checkpoint>_conf_N_imgsz_N_iou_N_timestamp/.
Training settings and source weights are recorded in experiment.json and args.yaml.
Old model paths used by train_apples.py and predict_apples.py are mapped through runs/layout_migration.json. External tools must use the new paths.
Example using the latest trained model:
```powershell
.\.venv\Scripts\python.exe predict_apples.py --baseline --conf 0.275 --imgsz 640 --iou 0.6
```

Directory naming update: dates now come first: timestamp_model_epN_szN_bN; counting: timestamp_checkpoint_conf_N_imgsz_N_iou_N. Existing weight paths are resolved through the migration map.
