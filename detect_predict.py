from ultralytics import YOLO
import cv2

if __name__ == '__main__': 

    # 加载训练完成的模型
    model = YOLO(r".\runs\detect\train2\weights\best.pt")

    # 读取测试图像
    test_path = r"datasets\board\images\test\15-1.jpg"
    test_img = cv2.imread(test_path)

    # 先展示一下是那一张图像
    cv2.imshow('Test Image', test_img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 使用该模型对图像执行对象检测
    results = model.predict(
        source=test_img,
        save=False,
        save_txt=False,
        show=True,
        conf=0.5,       # 调高置信度阈值 (默认0.25，增大可过滤低质量预测)
        iou=0.6,        # 调高IoU阈值 (默认0.45，增大使NMS更严格)
        max_det=300     # 限制最大检测数 (默认300，减少可限制重叠框数量)
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

