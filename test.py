
from ultralytics import YOLO
import cv2

if __name__ == '__main__': 

    # 重新生成客制化YOLO模型(一般来讲不推荐)
    # model = YOLO(r"models\v9\yolov9e.yaml")

    # 加载预训练的YOLO模型(推荐使用)
    model = YOLO(r"weights\yolov8n.pt")

    results = model('test1.jpg')  # list of 1 Results object

    # 使用 plot 方法绘制结果  
    plot_img = results.plot(draw_threshold=0.5)  # draw_threshold 是绘制框的置信度阈值  
    
    # 显示结果图像  
    cv2.imshow('Result Image', plot_img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    