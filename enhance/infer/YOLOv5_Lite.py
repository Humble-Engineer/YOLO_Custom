import cv2
import numpy as np
import onnxruntime as ort
import time

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1，c2分别表示框图的左上角和右下角的点
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

        
def _make_grid( nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs

def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):

    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)

    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]

def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):

    # 图像预处理
    img = cv2.resize(img0, [model_w,model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)

    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)

    return  boxes,confs,ids

# 构造对象储存检测目标的数据
class Data:
	
	# 输入分别为：查表后的类名，原始数据（x_min,y_min,x_max,y_max），图像尺寸（img_h,img_w）
	def __init__(self,name,data,shape):
		
		# 名称
		self.name = name
		
		# 绝对坐标及面积
		self.x = int((data[0]+data[2])/2)
		self.y = int((data[1]+data[3])/2)
		self.w = int(data[2]-data[0])
		self.h = int(data[3]-data[1])

		self.area = self.w*self.h
		
		# 相对坐标及面积
		self.x_rltv = 1.0*(data[0]+data[2])/(2*shape[1])
		self.y_rltv = 1.0*(data[1]+data[3])/(2*shape[0])
		self.w_rltv = 1.0*(data[2]-data[0])/shape[1]
		self.h_rltv = 1.0*(data[3]-data[1])/shape[0]

		self.area_rltv = (1.0*self.w*self.h)/(shape[0]*shape[1])
	
    # 打印数据结构体的各个属性
	def print(self):
		
		print ("\nclass_name: %s" %self.name)
		
		print ("x:%d, y:%d, w:%d, h:%d, area:%d" 
				%(self.x,self.y,self.w,self.h,self.area))
				
		print ("x:%0.2f, y:%0.2f, w:%0.2f, h:%0.2f, area:%0.2f" 
				%(self.x_rltv,self.y_rltv,self.w_rltv,self.h_rltv,self.area_rltv))
				

if __name__ == "__main__":

    # 模型加载
    model_pb_path = "./model/mul.onnx"  # 这里改为自己的onnx模型，注意用yolov5-lite中的export.py导出
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)
    
    # 标签字典
    dic_labels= {
        0:'duck',
        1:'box'
    }
    
    # 模型参数
    model_h = 320
    model_w = 320
    nl = 3
    na = 3
    stride=[8.,16.,32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    
    
    # 0表示从自带摄像头读取图像，1表示从usb摄像头读取图像，路径表示读取视频文件
    video = 1
    cap = cv2.VideoCapture(video)
    # False表示启动时不开启检测
    flag_det = False

    print("按下S键开始检测...")

    while True:

        # 返回视频流的读取结果和一帧图像
        success, img0 = cap.read()

        # 如果成功读取图像则开始处理
        if success:

            # 如果已经开启检测的话
            if flag_det:

                # 记录开始处理时的时间
                t1 = time.time()
                # 调用检测函数，并返回图框、置信度、代表类别的数字
                det_boxes,scores,ids = infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5)
                # 记录处理完成后的时间
                t2 = time.time()
                
                # 基于所记录的时间差计算运行帧率
                str_FPS = "FPS: %.2f"%(1./(t2-t1))
                
                # 将帧数也显示在图像上
                cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            
                # 依次对检测到的所有对象进行画框
                for box,score,id in zip(det_boxes,scores,ids):

                    # 生产标签，基于代表类别的数字查字典获得识别结果名称，并附上置信度
                    label = '%s:%.2f'%(dic_labels[id],score)
                    # 基于图框绘制检测后的图像
                    plot_one_box(box.astype(np.int16), img0, color=(255,0,0), label=label, line_thickness=None)
                    
                    # 输入原始数据,计算目标属性
                    obj = Data(dic_labels[id],box.astype(np.int16),img0.shape)
                    # 打印属性
                    # obj.print()
                    
            # 展示处理完成后的图像
            cv2.imshow("video",img0)

        # 等待键盘命令
        key=cv2.waitKey(1) & 0xFF  

        # Q键退出程序
        if key & 0xFF == ord('q'):
            print("程序已退出！")
            break

        # S键开始检测
        elif key & 0xFF == ord('s'):
            flag_det = not flag_det

            if flag_det:
                print("开始检测...")
            else:
                print("停止检测...")

        # C键截图保存
        elif key & 0xFF == ord('c'):
            
            # 获取当前时间
            t = time.localtime()

            # 生成当前日期字符串
            Capture_Day = str(t.tm_year)+"."+str(t.tm_mon)+"."+str(t.tm_mday)
            # 生成当前时间字符串
            Capture_Time = str(t.tm_hour)+"."+str(t.tm_min)+"."+str(t.tm_sec)

            # 设置截取图片名称
            Picture_Name = Capture_Day+"_"+Capture_Time
            # 设置截取图片所在路径
            Capture_Path = "./capture/"+Picture_Name+".jpg"
            
            # 保存截图
            cv2.imwrite(Capture_Path, img0)

            print("截图已保存:%s" % Capture_Path)

    # 释放摄像头
    cap.release() 
    # 摧毁所有窗口
    cv2.destroyAllWindows()
    

