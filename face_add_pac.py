import os
import shutil
import cv2
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# 创建文件夹包含了图片的数据
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
            size_img = cv2.resize(gray_img[y:y + h, x:x + w], (92, 112))
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
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
        # 关闭销毁窗口
    face.release()
    cv2.destroyAllWindows()
# 获取到文件夹下的所有图片信息
def Get_img_data(data_Img):
    data_img = []
    # 把某个文件夹里面的所有照片都遍历并且添加到data_img=[]中
    for file in os.listdir(data_Img):
        data_img.append(data_Img + '/' + file)
    return data_img
# 读取所有的人脸文件夹
def Get_document():
    face_list = []
    image = []
    label = []
    #遍历
    for i in range(1, 44):
        # 真~获取img+文件夹+ .jpg
        data_img = Get_img_data('./img/s' + str(i))
        for j in data_img:
            # 把完整的文件夹加图片添加到列表中
            face_list.append(j)
    # 读取列表数据,生产列表标签
    # enumerate的方法是将可遍历数据对象组合成一个索引序列
    for index, face in enumerate(face_list):
        # 读取图像的完整路径
        imgs = cv2.imread(face, 0)
        image.append(imgs)
        label.append(int(index / 10))
    print('文件夹里图片的个数：',len(image))
    print(label)
    return image,label,face_list
def data():
    # 把图像数据扁平化
    image_data = []
    image,label,face_list=Get_document()
    for images in image:
        data = images.flatten()
        image_data.append(data)
    print('一张图片的数据维度：',image_data[0].shape)
    # 转换为np数组
    X = np.array(image_data)
    y = np.array(label)
    print('转换为np数组后的维数：',X.shape)
    return X,y
# 划分训练集和测试集
def train_data():
    X,y=data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # pca降低维度
    pca = PCA(n_components=100)
    pca.fit(X_train)
    # 打印降维后的形态
    x_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)
    print('降到了100维后的数据维数X_train：',x_train_pca.shape)
    print('降到了100维后的数据维数X_test：',x_test_pca.shape)
    # 查看降维后的特征是所携带的原始数据的多少
    prefe = pca.explained_variance_ratio_.sum()
    print('降维后保留的原始特征数：', prefe)
    #交叉验证
    svc=SVC(kernel='linear')
    scores=cross_val_score(svc,x_train_pca,y_train,cv=4)
    print('交叉验证的平均分为：',scores.mean())
    #获取训练后的数据
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(x_train_pca, y_train)
    res = model.predict(x_test_pca[0])
# 预测准确率
    predicts = []
    num = 0
    for i in range(len(y_test)):
        res = model.predict(x_test_pca[i])
        if y_test[i] == res[0]:
            num += 1
    print('准确率为：{:.2f}%'.format(num / len(y_test)*100))
    return pca,model
#加载图片
#Get_img('./img')
def predict_img(document_img):
    pca,model=train_data()
    img1=cv2.imread(document_img)
    #修改尺寸
    #转换灰色图
    img2=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #重复类似上面的操作，扁平化,np,降维,预测....
    imgs=[]
    imgs.append(img2)
    img_data=[]
    for img in imgs:
        data=img.flatten()
        img_data.append(data)
    ret=np.array(img_data)
    print(ret.shape)
    test=pca.transform(ret)
    print(test[0].shape)
    res=model.predict(test)
    print('预测的结果为与第{:}个文件夹里面的人物最为相似。。。'.format(res[0]))

if __name__=="__main__":
    #Get_img('./img')
    predict_img('./img/s42/0.jpg')











