# original_codes
# 仓库简介
本开源仓库整合了多套自研深度学习完整实现代码，覆盖图像分割、实例检测、图像配准、图像分类、消融实验、对比实验、通用数据处理流水线等模块，全部为原创工程代码，标准化整理，适用于论文复现、学术实验、二次开发。
仓库地址：https://github.com/westbchampion/original_codes
仓库目录结构
```plaintext
original_codes/
├── Ghost-carafe-Unet     # 融合Ghost卷积与CARAFE上采样的改进UNet分割网络
├── Mask R-CNN            # 完整Mask R-CNN实例分割工程实现
├── Photo_registration    # 图像特征匹配与配准工具集
├── ablation_experiments  # 通用消融实验自动化运行框架
├── compae_experiment     # 算法对比实验基线代码
├── photo_classification  # 图像分类基础模型工程
├── process_pipeline      # 统一数据预处理+训练推理通用流水线
└── README.md             # 项目说明文档
```
各子模块详细介绍
1. Ghost-carafe-Unet
基于 UNet 改进的轻量化分割模型，适配遥感图像、医学影像分割任务：
引入 Ghost 卷积，大幅降低参数量与计算开销；
使用 CARAFE 内容感知上采样替换传统插值，还原精细边缘分割效果；
支持二分类 / 多分类分割，内置 Dice、交叉熵、Focal 等多种损失函数。
2. Mask R-CNN
基于 PyTorch 完整复现的实例分割工程：
包含 ResNet+FPN 骨干网络、RPN 候选框、ROIAlign、检测头 + 掩码头全套流程；
兼容 COCO、VOC 格式自定义数据集，自带训练、评估、结果可视化脚本。
3. Photo_registration
图像配准工具，适用于水下图像、遥感影像匹配对齐：
集成 SIFT、ORB、SuperPoint 主流特征提取匹配算法；
实现单应性矩阵求解、图像形变对齐、匹配效果可视化工具。
4. photo_classification
通用图像分类基线工程，轻量 / 重型骨干齐全：
内置 ResNet、MobileNet、EfficientNet 等主流分类网络；
自带数据增强、标签平滑、分类指标计算、训练日志记录功能。
5. ablation_experiments
通用消融实验自动化框架：
通过配置文件一键切换网络模块、损失函数、超参，无需重复改代码；
自动保存训练曲线、测试指标、对比图表，方便论文绘图。
6. compae_experiment
公平对比实验标准化代码：
统一训练、评估逻辑，保证不同算法实验条件完全一致；
批量运行脚本，自动输出汇总指标表格，直接用于论文撰写。
7. process_pipeline
全任务通用数据预处理流水线：
支持图像裁剪、缩放、归一化、标签转换、数据集划分；
可对接仓库内所有分割、检测、分类子项目，统一数据处理标准。
环境依赖
```python
运行
# 核心依赖包
python >= 3.8
torch >= 1.10
torchvision >= 0.11
opencv-python
numpy
scipy
matplotlib
tqdm
pillow
tensorboard
scikit-image
```
一键安装依赖：
```bash
运行
pip install -r requirements.txt
快速使用教程
1. 拉取仓库
```
```bash
运行
git clone https://github.com/westbchampion/original_codes.git
cd original_codes
```
引用说明
如果你在学术论文、项目中使用本仓库代码，欢迎点亮 Star，并按如下格式引用：
plaintext
@code{westbchampion2024originalcodes,
  title={original_codes: 深度学习原创算法代码库},
  author={westbchampion},
  year={2024},
  publisher={GitHub},
  url={https://github.com/westbchampion/original_codes}
}

开源协议
MIT 开源协议，允许免费用于学术研究、非商业项目；商业使用请联系作者授权。
交流与维护
代码 bug、参数疑问、功能需求可提交 Issues；
欢迎提交 Pull Request 优化代码、新增算法模块；
仓库最后更新时间：2024 年 11 月 12 日
