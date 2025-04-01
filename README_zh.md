# Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space
### <div align="center"> Yi Liu, Wengen Li*, Jihong Guan*, Shuigeng Zhou, Yichao Zhang <div>
### <div align="center"> CVPR 2025 <div>
<div align="center">
    <a href="https://github.com/Ly403/EMRDM"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
    <a href="./README.md"><img src="https://img.shields.io/static/v1?label=Readme&message=English&color=blue&logo=github-pages"></a> &ensp;
    <a href="https://arxiv.org/abs/2503.23717"><img src="https://img.shields.io/static/v1?label=Arxiv&message=EMRDM&color=red&logo=arxiv"></a> &ensp;
    <img src="https://img.shields.io/github/stars/Ly403/EMRDM?style=flat&logo=github"> &ensp;
    <a href="https://github.com/Ly403/EMRDM/issues"> <img src="https://img.shields.io/github/issues/Ly403/EMRDM?style=flat&logo=github"> </a> &ensp;
    <a href="https://github.com/Ly403/EMRDM/pulls"> <img src="https://img.shields.io/github/issues-pr/Ly403/EMRDM?style=flat&logo=github"> </a> &ensp;
</div>

![The cloud removal results](assets/visual_results.svg "cloud removal results")

这是**EMRDM**的官方实现仓库。EMRDM是一种新型的去云方法，其基于改进的均值回归的扩散模型。相比于以往的均值回归扩散模型，EMRDM拥有模块化的架构，该架构提供了一个灵活而清晰的设计空间，该架构的模块也经过重新优化设计。经过这些改进，EMRDM在单时态和多时态去云任务的公开数据集上都取得了SOTA的效果。

## :newspaper:新闻
- :dart:我们发布了EMRDM的官方实现；
- :dart:我们发布了在CUHK-CR1，CUHK-CR2，SEN12MS-CR和Sen2\_MTC\_New四个数据集上训练的EMRDM模型的权重； 
- :dart:我们提供了一份中文的readme文件；
- :dart:我们在本仓库提供了引用我们的文章的bibtex范例。

## :tada:项目的使用方法
### :wrench:环境配置
我们在本项目的根目录下面提供了[`requirements.txt`](./requirements.txt)，我们在虚拟环境里面使用的所有包的版本都写在了这个文件里面，但是不推荐直接用`pip install -r requirements.txt`从这个文件里面安装所有的包，因为这些包的依赖关系非常复杂，必须按照一定的顺序安装，且这个文件里面写的一些包是冗余的。推荐按照我们下面提供的详细步骤来安装包。

首先我们推荐安装一些比较大的且比较关键的包，例如`torch`，`flash_attn`，`natten`和`pytorch-lightning`。在安装这些包之前，可以使用`conda`来管理虚拟环境，`conda`创建虚拟环境的命令是：
```bash
conda create --name emrdm python=3.10
conda activate emrdm
```

然后，使用下面的命令来安装`torch`，顺便指定一下`numpy`的版本，把`numpy`也下载下来：
```bash
pip install torch==2.2.1 torchaudio==2.2.1 torchvision==0.17.1 numpy==1.26.4
```
需要注意的是，我们使用的`CUDA`版本是`CUDA 12.1`，这个版本信息可能在安装`torch`的时候有用。

接下来可以用下面的命令安装`flash_attn`：
```bash
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
```
但是其实这条命令会从头编译`flash_attn`，非常消耗时间。为了加速安装，可以从[`flash_attn`的官方仓库](https://github.com/Dao-AILab/flash-attention)下载一下wheel文件（注意是`2.5.9.post1`版本的wheel文件）然后使用下面的命令从wheel文件安装包：
```bash
pip install [name_of_flash_attn_wheel].whl
```

之后，要下载`natten`包，可以用如下命令：
```bash
pip install natten==0.17.1
```
同样这条命令会从头编译`natten`，很耗时间。你也可以到[`natten`的官方仓库](https://github.com/SHI-Labs/NATTEN/)下载wheel文件来加速。但是`natten`官方将他们的wheel文件放到了一个网站上，所以你可以通过`pip -f`指定wheel所在的网站来进行`natten`的下载，如下：
```bash
pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels
```

下载`pytorch-lightning`，可以使用如下这条命令：
```bash
pip install pytorch-lightning==2.3.0
```

在上面这些关键的包下载完之后，你可以用下面这些命令下载其他的辅助的包：
```bash
pip install wandb==0.17.8 matplotlib==3.9.2 natsort==8.4.0 \
omegaconf==2.3.0 scipy==1.14.0 dctorch==0.1.2 rasterio==1.3.11 
pip install pandas==2.2.3 opencv-python==4.10.0.84 lpips==0.1.4 
pip install tifffile==2024.7.24 s2cloudless==1.7.2 \
albumentations==1.4.10 albucore==0.0.12
```

（可选）如果你还是碰到了包丢失的问题，你可以参考[`requirements.txt`](./requirements.txt)文件来下载你缺失的包，这个文件里面提供了我们使用的虚拟环境里面所有包的版本信息。

**如果你还遇到了其他的上面没提到的环境配置问题，请和我们联系并将你遇到的问题报告给我们，或者在github上发一个issue。**
### :pushpin:数据集
我们使用了四个数据集：CUHK-CR1，CUHK-CR2，SEN12MS-CR和Sen2\_MTC\_New。你需要先把这些数据集下载好。

下面我们提供了下载这些数据集的网址：

|数据集|类型| 网址 |
|-------|----|-----|
|CUHK-CR1|单时态| [https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal) |
|CUHK-CR2|单时态| [https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal) |
|SEN12MS-CR|单时态| [https://patricktum.github.io/cloud_removal/sen12mscr/](https://patricktum.github.io/cloud_removal/sen12mscr/)|
|Sen2\_MTC\_New|多时态|[https://github.com/come880412/CTGAN](https://github.com/come880412/CTGAN)|

如果你想快点把代码跑起来，你可以先下载一下测试数据集，然后跑下面提供的测试命令。
### :mag_right:代码运行配置
在`./configs/example_training/`路径下面，我们提供了配置文件，也就是`*.yaml`文件。代码会自动读取`yaml`文件并设置代码运行的配置信息。你可以在`yaml`文件中修改这些配置，例如数据路径，batch size，读取数据的worker数目这些数据集的配置可以在`yaml`文件里面`data`这部分修改。如果想知道更详细的配置方法，可以看一下`./configs/example_training/`路径下面提供的范例`yaml`文件

我们也在`./configs/example_training/ablation/`路径下提供了我们进行消融实验时使用的配置文件，如果你有兴趣可以看一看。
### :fire:训练
在本项目的根路径下，使用下面的命令可以进行模型的训练：
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32
```
我们在`./configs/example_training/`提供了四个数据集训练的配置文件，[`cuhk.yaml`](./configs/example_training/cuhk.yaml)是用来在CUHK-CR1上训练的配置文件，[`cuhkv2.yaml`](./configs/example_training/cuhkv2.yaml)是用来在CUHK-CR2上训练的配置文件，[`sen2_mtc_new.yaml`](./configs/example_training/sen2_mtc_new.yaml) 是用来在Sen2\_MTC\_New数据集上训练的配置文件，[`sentinel.yaml`](./configs/example_training/sentinel.yaml)是用来在SEN12MS-CR数据集上训练的配置文件。注意：你需要修改一下这些`yaml`文件里面`data.params.train`部分的内容来适应你自己的数据集的配置（例如修改数据所在路径）。

你也可以在命令行上用`-l`参数来改变训练过程中日志文件的保存位置，不修改的话`./logs`就是默认的保存位置：
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -l [path_to_your_logs]
```

如果你想从一个之前训练好的检查点（checkpoint）继续训练，你可以使用命令行参数`-r`指定检查点的位置，如下：
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -r [path_to_your_ckpt]
```
如果你想从一个已训练好的检查点初始化模型，但是重启训练过程（即不从上次训练的中止epoch继续训练），你可以将`yaml`文件中的`model.ckpt_path`的值改为你的检查点的路径。
### :runner:Test
测试使用下面的命令：
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false
```
`[yaml_file_name].yaml`可以使用和训练过程中一样的`yaml`文件，但是需要做如下一些修改：
- 你需要设置`yaml`里面的`data.params.test`部分，以适配你的测试数据集的信息，否则测试集的dataloader将不会被实现，测试过程也就不会进行。
- 你需要修改`yaml`里面`model.ckpt_path`的值，将之改为你训练好的（或者是本项目提供的）检查点的路径。我们提供了一些已经训练好的EMRDM的检查点，可用于测试，请参考[模型](#model)一节。
### :computer:预测
预测过程会输出所有预测的去云图像（确保你有足够的硬盘空间），这个过程目前只支持单GPU运行，你需要修改`yaml`文件中的`lightning.trainer.devices`部分，只能设置一个设备。然后通过下面的命令运行预测过程：
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false --no-test true --predict true
```
`[yaml_file_name].yaml`和测试过程使用的是一样的。注意你还需要设置一下`yaml`文件里面的`data.params.predict`部分以及`model.ckpt_path`的值，设置方法和训练过程一样，不设置的话将无法得到正确结果。

## :open_file_folder:模型 {#model}
我们公布了所有的训练结果，包括训练的配置文件，训练得到的模型权重，还有训练过程的日志文件。

我们也公布了所有的测试结果，包括测试的配置文件，测试过程的日志文件。


所有的文件都可以通过下表展示的网址来下载：

|谷歌云盘|阿里云盘|百度云盘|
|------------|------------|-----------|
|https://drive.google.com/drive/folders/1T3OwRNP5r5qVLQZujnl2WDBVXHC1Am65?usp=sharing|https://www.alipan.com/s/39BcJezgsBC|https://pan.baidu.com/s/1RqYgluNNcYKXOa33kQioMQ|

在这些共享文件里面，`train`文件夹下放的是训练结果，`test`文件夹下放的是测试结果。百度云盘的分享码是*6161*。

## :sparkles:致谢
The code is based on the official implementations of the `generative-models`, `k-diffusion`, `utae-paps` and other repositories, as follows:
本仓库的代码主要基于`generative-models`，`k-diffusion`，`utae-paps`和其他一些以往的开源仓库的实现，我们将这些开源仓库和我们的项目的关系总结如下：

|仓库|网址|和本仓库的关系|
|----------|---|-------------------------------|
|`generative-models`|https://github.com/Stability-AI/generative-models|我们的代码的核心架构是基于该仓库。|
|`k-diffusion`|https://github.com/crowsonkb/k-diffusion|我们的代码中的去噪神经网络的实现是基于该仓库。|
|`utae-paps`|https://github.com/VSainteuf/utae-paps|我们的时序融合注意力机制的实现是基于该仓库。|
|`UnCRtainTS`|https://github.com/PatrickTUM/UnCRtainTS|我们代码中的SEN12MS-CR数据集的dataloader的实现是基于该仓库。|
|`SEN12MS-CR-TS`|https://github.com/PatrickTUM/SEN12MS-CR-TS|我们代码中的SEN12MS-CR数据集的dataloader的实现是基于该仓库。|
|`CTGAN`|https://github.com/come880412/CTGAN|我们代码中的Sen2\_MTC\_New数据集的dataloader的实现是基于该仓库。|
|`DDPM-Enhancement-for-Cloud-Removal`|https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal|我们代码中的CUHK-CR1和CUHK-CR2数据集的dataloader的实现是基于该仓库。|

我们向这些仓库的开发者致以诚挚的感谢，感谢他们对开源社区做出的巨大贡献，没有他们的项目就不会有本仓库。

## :email:联系方式
如果你遇到了任何问题，任何时候都可以通过下面两个邮箱联系我：<a href="mailto:liuyi2052697@foxmail.com">liuyi2052697@foxmail.com</a> (推荐)和<a href="mailto:liuyi61@tongji.edu.cn">liuyi61@tongji.edu.cn</a>。只要我有时间，我都会尽力回答你的问题。

## :book:引用方法
```bibtex
@misc{liu2025effectivecloudremovalremote,
      title={Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space}, 
      author={Yi Liu and Wengen Li and Jihong Guan and Shuigeng Zhou and Yichao Zhang},
      year={2025},
      eprint={2503.23717},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23717}, 
}
```