# Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Space
### <div align="center"> Yi Liu, Wengen Li$^{\ast}$, Jihong Guan$^{\ast}$, Shuigeng Zhou, Yichao Zhang <div>
### <div align="center"> CVPR 2025 <div>
<div align="center">
    <a href="https://github.com/Ly403/EMRDM"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
    <a href="./README_zh.md"><img src="https://img.shields.io/static/v1?label=项目简介&message=简体中文&color=blue&logo=github-pages"></a> &ensp;
    <a href="https://arxiv.org/abs/2503.23717"><img src="https://img.shields.io/static/v1?label=Arxiv&message=EMRDM&color=red&logo=arxiv"></a> &ensp;
    <img src="https://img.shields.io/github/stars/Ly403/EMRDM?style=flat&logo=github"> &ensp;
    <a href="https://github.com/Ly403/EMRDM/issues"> <img src="https://img.shields.io/github/issues/Ly403/EMRDM?style=flat&logo=github"> </a> &ensp;
    <a href="https://github.com/Ly403/EMRDM/pulls"> <img src="https://img.shields.io/github/issues-pr/Ly403/EMRDM?style=flat&logo=github"> </a> &ensp;
</div>

![The cloud removal results](assets/visual_results.svg "cloud removal results")
This is the official repository for **EMRDM**, which is a novel cloud removal methods improved from mean-reverting diffusion models, with a modular framework, a well-elucidated design space, and enhanced modules, demonstrating superior capabilities in both mono-temporal and multi-temporal cloud removal tasks.

## :newspaper:News
- :dart:We have released the implementation of EMRDM.
- :dart:We have made the weights of EMRDM trained on CUHK-CR1, CUHK-CR2, SEN12MS-CR, and Sen2\_MTC\_New datasets public.
- :dart:We have provided a readme file in Chinese.
- :dart:We have added a bibtex example for citing our work.

## :tada:Usage
### :wrench:Setup
We provide the [`requirements.txt`](./requirements.txt) in the root path. All the packages in the virtual environment we used are detailed in this file. But it is not recommended to directly use `pip install -r requirements.txt` as there are complex dependencies among these packages. We suggest to download some large and pivotal packages, like `torch`, `flash_attn`, `natten`, `pytorch-lightning` first. 
Specifically, it is highly suggested to first create a virtual enviroment using conda, as follows:
```bash
conda create --name emrdm python=3.10
conda activate emrdm
```

Then, you can download `numpy` and `torch` as follows:
```bash
pip install torch==2.2.1 torchaudio==2.2.1 torchvision==0.17.1 numpy==1.26.4
```
The version of `CUDA`  we used is `CUDA 12.1`. 

Next, you can download `flash_attn` as follows:
```bash
MAX_JOBS=4 pip install flash_attn==2.5.9.post1 --no-build-isolation
```
It may take a lot of time to build and compile the `flash_attn` package. For accelerating, you can also download the wheel from [the official repository of `flash_attn`](https://github.com/Dao-AILab/flash-attention) and use the following instruction to install from the wheel:
```bash
pip install [name_of_flash_attn_wheel].whl
```

After that, you can download the neighborhood attention package, *i.e*., `natten`, as follows:
```bash
pip install natten==0.17.1
```
This may also take a lot of time for compilation. Thus, you can also use the wheel from [the offficial repository of `natten`](https://github.com/SHI-Labs/NATTEN/) for fast downloading. Or you can directly use the following instrcution to download `natten` from its wheel:
```bash
pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels
```

For downloading `pytorch-lightning`, you can use the following instruction:
```bash
pip install pytorch-lightning==2.3.0
```

After downloading all the aforementioned pivotal packages, you can use the following instructions to download other packages:
```bash
pip install wandb==0.17.8 matplotlib==3.9.2 natsort==8.4.0 \
omegaconf==2.3.0 scipy==1.14.0 dctorch==0.1.2 rasterio==1.3.11 
pip install pandas==2.2.3 opencv-python==4.10.0.84 lpips==0.1.4 
pip install tifffile==2024.7.24 s2cloudless==1.7.2 \
albumentations==1.4.10 albucore==0.0.12
```

(*Optional*) If you still encounter the package missing problem, you can refer to the [`requirements.txt`](./requirements.txt) file to download the packages you need. 

**If you meet other enviroment setting problems we have not mentioned in this readme file, please contack us to report you problems or throw a issue.**
### :pushpin:Dataset
We use four datasets: CUHK-CR1, CUHK-CR2, SEN12MS-CR and Sen2\_MTC\_New. You need to download these datasets first. 

We provide the downloading URLs of these datasets as follows:

|Dataset|Type| URL |
|-------|----|-----|
|CUHK-CR1|Mono-Temporal| [https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal) |
|CUHK-CR2|Mono-Temporal| [https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal) |
|SEN12MS-CR| Mono-Temporal| [https://patricktum.github.io/cloud_removal/sen12mscr/](https://patricktum.github.io/cloud_removal/sen12mscr/)|
|Sen2\_MTC\_New| Multi-Temporal|[https://github.com/come880412/CTGAN](https://github.com/come880412/CTGAN)|

For fast starting, you can only download the testing dataset and run the testing instructions given below.

### :mag_right:Configurations
We provide our configuration files, *i.e.*, `*.yaml`, in the `./configs/example_training/` folder. The code automatically reads the `yaml` file and sets the configuration. You can change the settings, such as the data path, batch size, number of workers, in the `data` part of each `yaml` file to adapt to your expectations. Read the `yaml` file in `./configs/example_training/` for more details. 

We have also included the `yaml` files for our ablation experiments in the `./configs/example_training/ablation/` directory. 
### :fire:Train
You can use the following instruction in the root path of this repository to run the training process:
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32
```
Here, [`cuhk.yaml`](./configs/example_training/cuhk.yaml) is for training on the CUHK-CR1 dataset, [`cuhkv2.yaml`](./configs/example_training/cuhkv2.yaml) is for training on the CUHK-CR2 dataset, [`sen2_mtc_new.yaml`](./configs/example_training/sen2_mtc_new.yaml) is for training on the Sen2\_MTC\_New dataset, and [`sentinel.yaml`](./configs/example_training/sentinel.yaml) is for training on the SEN12MS-CR dataset. Note that you should modify the `data.params.train` part in the `yaml` file according to your dataset path.

You can also use the `-l` parameter to change the save path of logs, with `./logs` as the default path:

```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -l [path_to_your_logs]
```
If you want to resume from a previous training checkpoint, you can use the follow instruction:
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -r [path_to_your_ckpt]
```
If you want to initiate the model from an existing checkpoint and restart the training process, you should modify the value of `model.ckpt_path` in your `yaml` file to the path of your checkpoint.
### :runner:Test
Run the following instruction for testing:
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false
```
The `[yaml_file_name].yaml` files are the same as those in training process. Note that
- You should set the `data.params.test` part, otherwise the test dataloader will not be implemented.
- You should modify he value of `model.ckpt_path` in your `yaml` file to the path of your checkpoint. We have provided checkpoints of EMRDM trained on the four datasets used by us in the Sec. [Models](#model).
### :computer:Predict
The predicting process will output all cloud removed images. This process only support using one GPU (by setting `lightning.trainer.devices` to only one device). You can run predicting process using:
```bash
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false --no-test true --predict true
```
The `[yaml_file_name].yaml` files are the same as those in the testing process. Note that you should set the `data.params.predict` part and the `model.ckpt_path` part (the same way as testing), otherwise you will not obtain the correct results.

## :open_file_folder:Models {#model}
We have released all the training results, including the training configuration files, the weights of trained models, and the log files of the training process.

We have also released all the testing results, including the testing configuration files and the log files of the testing process.

All the files can be downloaded via the following URLs:

|Google Drive|Aliyun Drive|Baidu Drive|
|------------|------------|-----------|
|https://drive.google.com/drive/folders/1T3OwRNP5r5qVLQZujnl2WDBVXHC1Am65?usp=sharing|https://www.alipan.com/s/39BcJezgsBC|https://pan.baidu.com/s/1RqYgluNNcYKXOa33kQioMQ|


In the shared files, the `train` folder contains the training results, while the `test` folder contains the testing results. The sharing code for Baidu Drive is *6161*.

## :sparkles:Acknowledgment
The code is based on the official implementations of the `generative-models`, `k-diffusion`, `utae-paps` and other repositories, as follows:

|Repository|URL|Relationship to this Repository|
|----------|---|-------------------------------|
|`generative-models`|https://github.com/Stability-AI/generative-models|The main body of our code is based on this repository.|
|`k-diffusion`|https://github.com/crowsonkb/k-diffusion|The implementation of our denoising network is based on this repository.|
|`utae-paps`|https://github.com/VSainteuf/utae-paps|Our temporal fusion attention mechanism is based on this repository.|
|`UnCRtainTS`|https://github.com/PatrickTUM/UnCRtainTS|Our dataloader of the SEN12MS-CR dataset is based on this repository.|
|`SEN12MS-CR-TS`|https://github.com/PatrickTUM/SEN12MS-CR-TS|Our dataloader of the SEN12MS-CR dataset is based on this repository.|
|`CTGAN`|https://github.com/come880412/CTGAN|Our dataloader of the Sen2\_MTC\_New dataset is based on this repository.|
|`DDPM-Enhancement-for-Cloud-Removal`|https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal|Our dataloader of the CUHK-CR1 and CUHK-CR2 datasets are based on this repository.|

We express our sincere gratitude for their significant contributions to the open-source community. 

## :email:Contact me
If you have encountered any problems, feel free to contact me via my email <a href="mailto:liuyi2052697@foxmail.com">liuyi2052697@foxmail.com</a> (preferred) or <a href="mailto:liuyi61@tongji.edu.cn">liuyi61@tongji.edu.cn</a>. 

## :book:BibTeX
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