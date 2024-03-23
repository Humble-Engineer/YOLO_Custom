from ultralytics import YOLO

if __name__ == '__main__': 
    
    model = YOLO(r"weights\yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    data_path = r"datasets\coco128-seg.yaml"
    results = model.train(data=data_path, epochs=300, imgsz=640)