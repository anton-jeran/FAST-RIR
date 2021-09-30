# NN-DAS
This is the official implementation of our Real-time neural-network-based diffuse acoustic simulator. Our model is inspired from [**StackGAN**](https://github.com/hanzhanggit/StackGAN-v2) architecture.

## Requirements

'''
Python3
Pytorch
python-dateutil
easydict
pandas
torchfile
gdown
'''

## Generete RIRs using trained model

Download the trained model and embedding examples using this command

'''
source download_generate.sh
'''

Run the following command inside code_new to generate multi-channel RIRs corresponding to the embeedings. In our example each embedding contains 3 7-channel RIRs.

'''
python3 main.py --cfg cfg/RIR_eval.yml --gpu 0
'''

## Embedding

Each normalized embedding is created as follows:

'''
Listener Position = LP
Source Position = SP
Room Dimension = RD
Reverberation Time = T60

Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_y,SP_Z,RD_X,RD_Y,RD_Z,T60] /5) + 1
'''

You can find example embedding files in **/generate/embeddings** folder.

## Range

Our trained NN-DAS is capable of generating RIRs with following range accurately.

'''
Room Dimension X --> 8m to 11m
Room Dimesnion Y --> 6m to 7m
Room Dimension Z --> 2.5m to 3.5m
Listener Position --> Any position within the room
Speaker Position --> Any position within the room
Reverberation time --> 0.2s to 1s
'''

## Training the Model

Run following command to download the training dataset we created using a [**Diffuse Acoustic Simulator**](https://github.com/GAMMA-UMD/pygsound). You also can train the model using your own dataset.

'''
source download_data.sh
'''

Run the following command to train the model. You can pass what GPUs to be used for training as an input argument. In this example, I am using 2 GPUs.

'''
python3 main.py --cfg cfg/RIR_s1.yml --gpu 0,1
'''


## Related Works
1) [**IR-GAN: Room Impulse Response Generator for Far-field Speech Recognition (INTERSPEECH2021)**](https://github.com/anton-jeran/IR-GAN)
2) [**TS-RIR: Translated synthetic room impulse responses for speech augmentation (IEEE ASRU 2021)**](https://github.com/GAMMA-UMD/TS-RIR)


## Citations
If you use our NN-DAS for you research, please consider citing
'''
[**Coming Soon**]
'''

Our work is inspired from
'''
@inproceedings{han2017stackgan,
Author = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
Title = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks},
Year = {2017},
booktitle = {{ICCV}},
}
'''

If you use our training dataset generated using [**Diffuse Acoustic Simulator**] in your research, please consider citing
'''
@inproceedings{9052932,
  author={Z. {Tang} and L. {Chen} and B. {Wu} and D. {Yu} and D. {Manocha}},  
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Improving Reverberant Speech Training Using Diffuse Acoustic Simulation},   
  year={2020},  
  volume={},  
  number={},  
  pages={6969-6973},
}
'''

