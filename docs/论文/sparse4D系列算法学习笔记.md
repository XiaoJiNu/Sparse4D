# sparse4D系列算法学习笔记

## Sparse4D 算法框架

### 问题

#### 问题１：６个学习的关键点是什么

Deformable 4D Aggregation 模块中

#### 问题２：运动补偿怎么做的？

#### 问题3：**层级化特征融合**中*Fuse Multi-Scale/View*怎么做？

## Sparse4D-V2算法框架

### 问题

#### 问题1:Efficient Deformable Aggregation具体如何实现的？

#### 问题2:时序投影如何实现的？

## Sparse4D-v3算法框架

### 问题

#### 问题1: 3D检测加噪公式是如何实现的？

#### 问题2: 匈牙利匹配问题

one2one 匈牙利匹配过程中，正样本离GT并不能保证一定比负样本更近，而且正样本的分类loss并不随着匹配距离而改变。为什么？