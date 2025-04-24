from ultralytics import YOLO

if __name__ == '__main__': 

    # 重新生成客制化YOLO模型(一般来讲不推荐)
    # model = YOLO(r"models\v8\yolov8n.yaml")

    # 加载预训练的YOLO模型(推荐使用)
    model = YOLO(r"weights\yolo11n.pt")

    # 使用指定的数据集训练模型若干个epoch
    data_path = r"datasets\board.yaml"
    results = model.train(data=data_path, epochs=500, batch=-1)

    # 将模型导出为ONNX格式
    # success = model.export(format='onnx')
