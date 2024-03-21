
自定义的YOLO项目文件夹

datasets/    存放各种数据集

    <dataset name>/    数据集.yaml文件所在的文件夹


documents/    存放使用教程


models/    存放各个版本YOLO的.yaml构建文件


runs/    存放训练、推理过程中产生的文件

    predict/    存放推理结果

    train/  存放训练效果样张等结果

        weights/    存放训练产生的.pt权重文件


weights/    存放官方给出的预训练权重文件


train.py    训练程序
predict.py  推理程序