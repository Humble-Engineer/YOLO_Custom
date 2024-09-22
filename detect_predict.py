from ultralytics import YOLO
import cv2

if __name__ == '__main__': 

    # 加载训练完成的模型
    model = YOLO(r".\runs\detect\train2\weights\best.pt")

    # 读取测试图像
    test_path = r"datasets\hydrogel\images\val\neg_e2 (1).png"
    # test_path = r"test5.jpg"
    test_img = cv2.imread(test_path)

    # 先展示一下是那一张图像
    cv2.imshow('Test Image', test_img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 使用该模型对图像执行对象检测
    results = model.predict(source=test_img, save=False, save_txt=False, show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

