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


# TCLA

This repository is a simplified, GitHub-ready version of the smooth latent
alignment code. It keeps small Chewie and Mihili example datasets and the
minimal model code needed to train a smooth source autoencoder and run
conditional MMD target alignment.

## Contents

```text
TCLA/
├── README.md                                     
├── requirements.txt                              # Python dependencies
├── config/                                       # Example hyperparameter files
│   ├── behavioral_decoders.yaml                  # Ridge and LSTM decoder hyperparameters
│   ├── chewie_co_2016_cross_session.yaml         # Chewie session_0 -> session_1 experiment config
│   └── chewie_to_mihili_cross_subject.yaml       # Chewie session_0 -> Mihili session_0 experiment config
├── data/                                         # Small example datasets
│   ├── Chewie_CO_2016/
│   │   ├── session_0.pickle                      # Source session example
│   │   ├── session_1.pickle                      # Target session example
│   │   └── adan_channel_ids.json                 # ADAN channel mapping metadata
│   └── Mihili_CO_2014/
│       └── session_0.pickle                      # Cross-subject target session example
├── TCLA/                                         
│   ├── data/
│   │   └── data_load.py                          # Session pickle loader
│   ├── networks/
│   │   ├── blocks.py                             # Modules
│   │   ├── count_wrapper.py                      # AE output wrapper
│   │   └── s4.py                                 # S4 sequence blocks
│   └── losses.py                                 # Training losses
└── examples/                                     
    ├── Chewie_CO_2016/
    │   ├── AE_model/train.py                     # Stage1 source training
    │   ├── Stage2_MMD/with_align_all_labels.py   # Cross-session target alignment
    │   └── to_Mihili_CO_2014/
    │       └── Stage2_MMD/with_align_all_labels.py # Cross-subject target alignment
    └── behavioral_decoder/
        ├── ridge_decoder.py                      # Ridge decoder on Stage2 latents
        └── lstm_decoder.py                       # LSTM decoder on Stage2 latents
```

## Dependencies

To get start, we recommend creating a conda environment first.

```bash
git clone git@github.com:FAMD-CASIA/TCLA.git
cd TCLA
conda create --name TCLA python=3.9
conda activate TCLA
pip install -r requirements.txt
```

## Data Format
Save your data as .pickle format and should contain three keys:

```python
{
    "spike": spike_array,
    "behavior": behavior_array,
    "label": label_array,
}
```

Expected array shapes:

- `spike`: `num_trials x num_time_bins x num_neurons`
- `behavior`: `num_trials x num_time_bins x 2`
- `label`: `num_trials`

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
