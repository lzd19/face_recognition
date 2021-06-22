import os
import shutil
import cv2
import numpy as np
def Get_img(data):
    # 加载脸的xml文件,用于后面判断人脸
    face_xml = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # 用户保存的id，用于后续识别出图片中的人物
    user_id = input('用户的ID:')
    # 判断待会如果要保存的文件夹已经存在则删除该文件夹
    path = os.path.join(data, user_id)
    if os.path.isdir(path):

        shutil.rmtree(path)
    # 创建一个文件夹
    os.mkdir(path)
    # 开启视频获取人脸
    face = cv2.VideoCapture(0)
    # 设置要保存的图片的初始化
    number = 0
    while True:
        # ok里存放着是否读取到人脸的结果,img表示获取到人脸
        ok, img = face.read()
        # 转换成灰度图像，降低计算难度
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 存放脸在图片中的大小和图片与检测人脸
        faces = face_xml.detectMultiScale(
            gray_img,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(32, 32)
        )
        # 设置显示出来的视频带有美颜效果(没有美颜不能活）
        fanny_img = cv2.bilateralFilter(img, 30, 20 * 2, 20 / 2)
        # 把检测到的人脸用矩形给框起来
        for (x, y, w, h) in faces:
            # 设置图片大小
            size_img = cv2.resize(img[y:y + h, x:x + w], (92, 112))
            # 可以保存图片
            cv2.imwrite('%s/%s.jpg' % (path, str(number)), size_img)
            # 画出矩形
            cv2.rectangle(fanny_img, (x, y), (w + x, y + h), (0, 255, 0), 2)
            # 显示文字
            # cv2.putText(fanny_img, 'liu', (x + w + 5, y - 10), font, 3, (0, 0, 255), 3)
            # 显示图片
            number += 1
        cv2.imshow('face_test', fanny_img)
        # 设置延时，不然看不见
        # key = cv2.waitKey(5)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
        # 关闭销毁窗口
    face.release()
    cv2.destroyAllWindows()


def Train_img(data):
    # 创建列表存放信息
    lables = []
    face_list = []
    imgs = []
    label = 0

    for each_img in os.listdir(data):
        # 把each_img里的所有图片都连接到List_img
        List_img = os.path.join(data, each_img)
        # 判断List_img是否存在xinxi
        if os.path.isdir(List_img):
            # 把List_img的信息存放到face_list里
            face_list.append(each_img)
            # 遍历List_img文件里面的图片
            for face_img in os.listdir(List_img):
                imgpath = os.path.join(List_img, face_img)
                # 读取照片文件
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                # 转换图像为灰度图，降低计算机的计算难度
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 把图像文件放入上面创建好的列表里
                imgs.append(gray_img)
                lables.append(label)
            label += 1
        # 转换数据进行格式长度转换数据矩阵
    X = np.asarray(imgs)
    y = np.asarray(lables)
    return X, y, face_list


def True_tain_face(data):
    # 获取训练的数据
    X, y, names = Train_img(data)
    # 训练
    trains = cv2.face.EigenFaceRecognizer_create()
    trains.train(X, y)
    # 开启摄像头
    camera = cv2.VideoCapture(0)
    # cv2.namedWindow()
    face_xml = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    while True:
        ok, frame = camera.read()
        # 判断是否读取成功
        if ok:
            # 把读取到的图像进行灰度化，以便于下面的对比
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 鉴别人脸
            faces = face_xml.detectMultiScale(
                gray_img,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(32, 32)
            )
            # 画出矩形
            fanny_img = cv2.bilateralFilter(frame, 30, 20 * 2, 20 / 2)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(fanny_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                size_img = gray_img[y:y + h, x:x + w]
                # 调整大小且使用像素区域关系进行重采样
                size_img = cv2.resize(size_img, (92, 112), interpolation=cv2.INTER_AREA)
                # 预测
                predices = trains.predict(size_img)
                # 输出结果
                cv2.putText(frame, names[predices[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # 显示图像
            cv2.imshow('video', frame)
            # 判断推出条件
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data = './img'
    #Get_img(data)
    True_tain_face(data)
