# face_recognition
<h2>这是一个基于opencv的人脸识别项目</h2>

<div>在使用这个人脸识别功能之前，你需要安装能够运行该项目的环境</div>
<div>pip install opencv-python</div>
<div>pip install opencv-contrib-python</div>
<div>该项目下载到本地后需要在face-train-over.py的同级文件夹创建一个img的文件夹用于存放照片(当然了，你也可以创建其他名字的文件夹，只不过可能要修改某些地方)</div>
<div>关于opencv：这是一个是一个开源的计算机视觉库，它提供了很多函数，这些函数非常高效地实现了计算机视觉算法（最基本的滤波到高级的物体检测皆有涵盖），它的底层源码使用c++开发，不过它提供了有python的
api接口</div>
<div>本项目只是提供了非常简单的人脸识别功能，包括人脸的存取，训练，与检测,  如果对您有帮助请点个   小星星   ,如在使用的过程中遇到一些bug，欢迎issues!!!!我会不断地对问题进行改进...</div>
<div>项目参考文献1：OpenCV中文官方文档http://www.woshicver.com/</div>
<div>项目参考文献2：使用OpenCV和Python进行人脸识别https://www.cnblogs.com/zhuifeng-mayi/p/9171383.html</div>
<div>注：假如导入的照片已经拥有一定的数量，可以选择注释掉倒数第二行的  Get_img(data)  只进行人脸识别判断而不重新进行人脸保存</div>
