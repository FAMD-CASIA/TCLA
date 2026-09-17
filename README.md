# examples/Chewie_CO_2016文件夹中：

AE_model是Chewie_CO_2016数据集source session训练的代码

Stage2_MMD是Chewie_CO_2016数据集target session的cross-session代码

to_Mihili_CO_2014/Stage2_MMD是以Mihili_CO_2014数据集作为target session的cross-subject代码


# data文件夹中：

解压文件unzip_this_file.zip,其中包含Chewie_CO_2016和Chewie_CO_2016的两个数据集用到的示例数据

数据格式：

data['spike'] [number of trials, trial_length, number of  channels] 数据的spike信息

data['behavior'] [number of trials, trial_length, number of  channels] 数据的行为信息(position)

data['label'] [number of trials,] 行为标签，数值是[0,1,2,...7]中的一个

# TCLA文件夹中：

data是数据加载方式

network是用到的网络

losses是损失函数


test



# Introduction

This repository is a simplified, GitHub-ready version of the smooth latent
alignment code. It keeps small Chewie and Mihili example datasets and the
minimal model code needed to train a smooth source autoencoder and run
conditional MMD target alignment.

## Contents

- `TCLA/`: core Python package.
- `data/Chewie_CO_2016/`: two example sessions, `session_0.pickle` and
  `session_1.pickle`.
- `data/Mihili_CO_2014/`: one example target session, `session_0.pickle`.
- `examples/Chewie_CO_2016/AE_model/train.py`: smooth Stage1 source-session
  training with latent Gaussian smoothing, latent dynamics regularization, and
  behavior curvature loss.
- `examples/Chewie_CO_2016/Stage2_MMD/with_align_all_labels.py`: Stage2
  conditional MMD alignment for `session_0 -> session_1`, reusing the smooth
  Stage1 checkpoint.
- `examples/Chewie_CO_2016/to_Mihili_CO_2014/Stage2_MMD/with_align_all_labels.py`:
  cross-subject conditional MMD template for `Chewie_CO_2016/session_0` to
  `Mihili_CO_2014/session_0`, reusing the same smooth Stage1 checkpoint.

## Quick Check

Run a lightweight import/data/model-forward check:

```bash
python smoke_test.py
```

## Example Training

Stage1 source training:

```bash
NUM_EPOCHS=1 NUM_WARMUP_EPOCHS=0 NUM_WORKERS=0 \
python examples/Chewie_CO_2016/AE_model/train.py
```

Stage2 target alignment, after Stage1 has produced the source checkpoint:

```bash
TARGET_SESSION_ID=1 ADAPT_NUM_EPOCHS=1 ADAPT_NUM_WARMUP_EPOCHS=0 NUM_WORKERS=0 \
python examples/Chewie_CO_2016/Stage2_MMD/with_align_all_labels.py
```

Cross-subject Stage2 target alignment, after Stage1 has produced the source
checkpoint:

```bash
TARGET_SESSION_ID=0 ADAPT_NUM_EPOCHS=1 ADAPT_NUM_WARMUP_EPOCHS=0 NUM_WORKERS=0 \
python examples/Chewie_CO_2016/to_Mihili_CO_2014/Stage2_MMD/with_align_all_labels.py
```

Outputs are written under `outputs/`, which is ignored by git.
