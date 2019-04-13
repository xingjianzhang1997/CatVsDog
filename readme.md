## 文件说明
（1）data文件夹下包含test和train两个子文件夹，分别用于存放测试数据和训练数据，从官网上下载的数据直接解压到相应的文件夹下即可。  
（2）venv文件夹用于存放加载anaconda环境。  
（3）log文件用来保存训练结果和参数。  
 以上三个文件可以自己在本地新建。  
（4）input_data.py负责实现读取数据，生成批次（batch）。  
（5）model.py负责实现我们的神经网络模型。   
（6）training.py负责实现模型的训练以及评估。  
（7）test-1.py 从test文件中随机测试一张图片。  
（8）test-2.py 从test_image文件中测试一张图片，方便指定图像测试。  
（9）test-3.py 查看input_data.py读取并统一大小后的图片。  

## 改进方案：
（1）加入dropout机制，其是一种正则化（regularization）技术，用来来防止过拟合。  
（2）换卷积结构如VGG16。  
 (3) 用OpenCV对图片进行预处理。  

