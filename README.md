## vgg feature for image deblocking
### 1.Introduction  
&emsp;&emsp;相比于以前仅以有噪图像作为单一输入的网络模型，为进一步提高去块效果，尤其是增强主观视觉质量。参考：[One-to-Many Network for Visually Pleasing Compression Artifacts Reduction](https://arxiv.org/abs/1611.04994)的模型结构，增加通过vgg16 模型提取的特征作为另一路输入，进行训练。
  
### 2.Model
&emsp;&emsp;图像去块网络采用15个residual block。特征提取网络用[vgg16](https://arxiv.org/abs/1409.1556)第13层卷积层中的第10层卷积层的输出。  
&emsp;&emsp;
### 3.Train
&emsp;&emsp;1.数据集：H.265压缩，qp=40
&emsp;&emsp;2.训练：batchsize:64 epoch:100 optimizer:Adam(lr=0.0001)

### 4.Experiment Result
训练模型 | PSNR|SSIM|train loss
---|---|---|---|
input(qp=40) | 29.5838|0.8448|
resnet |30.0922|0.8556|0.00110
resnet+vgg |29.5986|0.8483|0.000977
ARCNN|30.0818|0.8552|0.00110
ARCNN+vgg|30.0260|0.8543|0.00110



### 5.Analysis&BUG
&emsp;&emsp; 1.一个明显的问题是模型训练的mse loss表现好，但验证和测试时loss表现差,并不是过拟合,这个尚未找到原因。

&emsp;&emsp; 2.数据预处理:vgg模型所用数据集的预处理是用imagenet的均值和方差进行归一化处理，而一般去块模型则只除以255作为简单处理。这里有一个问题，就是所用的特征提取网络所要求的图像预处理不同.

  
 依次为label input resnet resnet+vgg：  
![image](https://github.com/yydlmzyz/vgg-feature-for-image-deblocking/blob/master/images/laebl.PNG)  
![image](https://github.com/yydlmzyz/vgg-feature-for-image-deblocking/blob/master/images/input.PNG)  

![image](https://github.com/yydlmzyz/vgg-feature-for-image-deblocking/blob/master/images/resnet.PNG)  
![image](https://github.com/yydlmzyz/vgg-feature-for-image-deblocking/blob/master/images/resnet+vgg.PNG)  




