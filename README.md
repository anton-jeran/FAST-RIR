# FAST-RIR: FAST NEURAL DIFFUSE ROOM IMPULSE RESPONSE GENERATOR
This is the official implementation of our neural-network-based  fast  diffuse  room  impulse  response  generator (**FAST-RIR**)  for  generating  roomimpulse responses (RIRs) for a given rectangular acoustic environment. Our model is inspired by [**StackGAN**](https://github.com/hanzhanggit/StackGAN-Pytorch) architecture. The audio examples and spectrograms of the generated RIRs are available [here](https://anton-jeran.github.io/FRIR/).

## Requirements

```
Python3.6
Pytorch
python-dateutil
easydict
pandas
torchfile
gdown
pickle
```


## Embedding

Each normalized embedding is created as follows: If you are using our trained model, you may need to use extra parameter Correction(CRR).

```
Listener Position = LP
Source Position = SP
Room Dimension = RD
Reverberation Time = T60
Correction = CRR

CRR = 0.1 if 0.5<T60<0.6
CRR = 0.2 if T60>0.6
CRR = 0 otherwise

Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) + 1
```


## Generete RIRs using trained model

Download the trained model using this command

```
source download_generate.sh
```

Create normalized embeddings list in pickle format. You can run following command to generate an example embedding list
```
 python3 example1.py
```

Run the following command inside **code_new** to generate RIRs corresponding to the normalized embeddings list. You can find generated RIRs inside **code_new/Generated_RIRs**

```
python3 main.py --cfg cfg/RIR_eval.yml --gpu 0
```

## Range

Our trained NN-DAS is capable of generating RIRs with the following range accurately.
```
Room Dimension X --> 8m to 11m
Room Dimesnion Y --> 6m to 8m
Room Dimension Z --> 2.5m to 3.5m
Listener Position --> Any position within the room
Speaker Position --> Any position within the room
Reverberation time --> 0.2s to 0.7s
```

## Training the Model

Run the following command to download the training dataset we created using a [**Diffuse Acoustic Simulator**](https://github.com/GAMMA-UMD/pygsound). You also can train the model using your dataset.

```
source download_data.sh
```

Run the following command to train the model. You can pass what GPUs to be used for training as an input argument. In this example, I am using 2 GPUs.

```
python3 main.py --cfg cfg/RIR_s1.yml --gpu 0,1
```


## Related Works
1) [**IR-GAN: Room Impulse Response Generator for Far-field Speech Recognition (INTERSPEECH2021)**](https://github.com/anton-jeran/IR-GAN)
2) [**TS-RIR: Translated synthetic room impulse responses for speech augmentation (IEEE ASRU 2021)**](https://github.com/GAMMA-UMD/TS-RIR)


## Citations
If you use our **FAST-RIR** for your research, please consider citing

```
[**Coming Soon**]
```

Our work is inspired by
```
@inproceedings{han2017stackgan,
Author = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
Title = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks},
Year = {2017},
booktitle = {{ICCV}},
}
```

If you use our training dataset generated using [**Diffuse Acoustic Simulator**](https://github.com/GAMMA-UMD/pygsound) in your research, please consider citing
```
@inproceedings{9052932,
  author={Z. {Tang} and L. {Chen} and B. {Wu} and D. {Yu} and D. {Manocha}},  
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Improving Reverberant Speech Training Using Diffuse Acoustic Simulation},   
  year={2020},  
  volume={},  
  number={},  
  pages={6969-6973},
}
```

