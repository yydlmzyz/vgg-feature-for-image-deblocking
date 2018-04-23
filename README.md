## vgg feature for image deblocking
### 1.Introduction  
&emsp;&emsp;相比于以前仅以有噪图像作为单一输入的网络模型，为进一步提高去块效果，尤其是增强主观视觉质量。参考：[One-to-Many Network for Visually Pleasing Compression Artifacts Reduction](https://arxiv.org/abs/1611.04994)的模型结构，增加通过vgg16 模型提取的特征作为另一路输入，进行训练。
  
### 2.Model
&emsp;&emsp;图像去块网络暂用[L8](https://arxiv.org/pdf/1605.00366.pdf)。特征提取网络用[vgg16](https://arxiv.org/abs/1409.1556)。参考论文中用来构造损失函数的特征层是其13层卷积层中的第10层卷积层的输出。  
&emsp;&emsp;
### 3.Train
&emsp;&emsp;1.数据集：H.265压缩，qp=40
&emsp;&emsp;2.训练：batchsize:64 epoch:1000 optimizer:Adam(lr=0.0001)

### 4.Experiment Result
训练模型 | PSNR|SSIM
---|---|---|
input(qp=40) | 29.5838|0.8448|
pretrained model | 26.8431(+1.23)|0.7793(+0.040)
only feature loss*5 |25.7689(+0.16)|0.7304(-0.001)
only pixel loss*1|26.8752(+1.26)|0.7815(+0.042)
both pixel loss*1+ feature loss*5|26.5002(+0.9437)|0.7726(+0.033)



### 5.Analysis&BUG
#### 1.residual block:模型中所用残差块无BN层，因为有BN层，会导致去块效果不升反降，已经检查过model的train/eval模式设置，但还未找到原因。
#### 2.data pre-process:vgg模型所用数据集的预处理是用imagenet的均值和方差进行归一化处理，而一般去块模型则只除以255作为简单处理。这里有一个问题，就是所用的特征提取网络所要求的图像预处理不同。
#### 3.一个明显的问题是模型训练的mselo ss表现好，但验证和测试时loss表现差,这个也一直未找到原因。
&emsp;&emsp; 
  
  左图为用feature loss,右图为没有feature loss：  
![image](https://github.com/yydlmzyz/Feature-losses-for-image-deblocking/blob/master/images/compare.JPG)  
![image](https://github.com/yydlmzyz/Feature-losses-for-image-deblocking/blob/master/images/compare2.jpg)  
kua

