# Radiowave propagation simulation
2D Raidowave propagation simulation  based on the parabolic equation with Split-step Fourier transform(SSFT) and Intelligent propagation model

## 文件说明
* **SSFT_PE_2D.m**: 利用抛物方程的分布傅里叶算法计算大范围的电波传播路径损耗。
* **./ray**: 利用pharlap工具箱做的射线追踪和抛物方程的对比。
* **./deeplearning**: 利用pharlap提供的gen_iono_grid_2d函数和SSFT_PE_2D.m批量制作电子密度数据以及对应的电波传播损耗图像数据集，并利用深度学习模型进行训练。
