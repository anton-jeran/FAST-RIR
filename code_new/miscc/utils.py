import os
import errno
import numpy as np

from copy import deepcopy
from miscc.config import cfg
from scipy.io.wavfile import write
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from wavefile import WaveWriter, Format
import RT60
from multiprocessing import Pool


#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_RIRs, fake_RIRs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_RIRs.size(0)
    cond = conditions.detach()
    fake = fake_RIRs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_RIRs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    # real pairs
    #print("util conditions ",cond.size())
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data, errD_wrong.data, errD_fake.data
    # return errD, errD_real.data[0], errD_wrong.data[0], errD_fake.data[0]



def compute_generator_loss(epoch,netD,real_RIRs, fake_RIRs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    loss = nn.L1Loss() #nn.MSELoss()
    loss1 = nn.MSELoss()
    RT_error = 0
    # print("num", real_RIRs.size(),real_RIRs.size()[0])
    # input("kk")
   

    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_RIRs), gpus)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    MSE_error = loss(real_RIRs,fake_RIRs)
    MSE_error1 = loss1(real_RIRs,fake_RIRs)
    sample_size = real_RIRs.size()[0]
    channel = 12
    fs = 16000
    rn = np.random.randint(sample_size-(channel*2))
    real_wave = np.array(real_RIRs[rn:rn+channel].to("cpu").detach())
    real_wave = real_wave.reshape(channel,4096)
    fake_wave = np.array(fake_RIRs[rn:rn+channel].to("cpu").detach())
    fake_wave = fake_wave.reshape(channel,4096)

    pool = Pool(processes=12)
    
    results =[]
    for n in range(channel):
        results.append(pool.apply_async(RT60.t60_parallel, args=(n,real_wave,fake_wave,fs,)))
    
    T60_error =0
    for result in results:
        T60_error =  T60_error + result.get()

    RT_error = T60_error/channel
    
    pool.close()
    pool.join()
   
    
   
    
    # T60_error =0
    # for m in range(channel):
    #     real_wave_single   = real_wave[:,(rn+m)]
    #     fake_wave_single   = fake_wave[:,(rn+m)]
    #     Real_T60_val = RT60.t60_impulse(real_wave_single,fs)
    #     Fake_T60_val = RT60.t60_impulse(fake_wave_single,fs)
    #     T60_diff = abs(Real_T60_val-Fake_T60_val)
    #     T60_error =  T60_error + T60_diff

    # RT_error = T60_error/channel
    
    
    # r = WaveWriter("real.wav", channels=portion, samplerate=fs)
    # r.write(np.array(real_IR))
    # f = WaveWriter("fake.wav", channels=portion, samplerate=fs)
    # f.write(np.array(fake_IR))


    # result = call_python_version("3.8", "RT60", "t60_error",  
    #                          ["real.wav","fake.wav"])
    # # print("RT_error ",result)
    # RT_error = float(result)
   

    # print("RT_error ",RT_error)

    # if(epoch<100):
    #     errD_fake = criterion(fake_logits, real_labels)# + 2* 4096 * MSE_error
    # else:
    #     errD_fake = criterion(fake_logits, real_labels) + 2* 4096 * MSE_error
    errD_fake = criterion(fake_logits, real_labels) + 5* 4096 * MSE_error1 + 40 * RT_error
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake, MSE_error,RT_error


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_RIR_results(data_RIR, fake, epoch, RIR_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_RIR is changed to [0,1]
    if data_RIR is not None:
        data_RIR = data_RIR[0:num]
        for i in range(num):
            # #print("came 1")
            real_RIR_path = RIR_dir+"/real_sample"+str(i)+".wav" 
            fake_RIR_path = RIR_dir+"/fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000

            real_IR = np.array(data_RIR[i].to("cpu").detach())
            fake_IR = np.array(fake[i].to("cpu").detach())
            # #print("fake_IR ", fake_IR.size)
            # #print("real_IR ", real_IR.size)
            # #print("max real_IR ", max(real_IR[0]))
            # #print("min real_IR ", min(real_IR[0]))
            r = WaveWriter(real_RIR_path, channels=1, samplerate=fs)
            r.write(np.array(real_IR))
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(fake_IR))           


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)


            # write(real_RIR_path,fs,real_IR)
            # write(fake_RIR_path,fs,fake_IR)

        # vutils.save_image(
        #     data_RIR, '%s/real_samples.png' % RIR_dir,
        #     normalize=True)
        # # fake.data is still [-1, 1]
        # vutils.save_image(
        #     fake.data, '%s/fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)
    else:
        for i in range(num):
            # #print("came 2")
            fake_RIR_path = RIR_dir+"/small_fake_sample"+str(i)+"_epoch_"+str(epoch)+".wav"
            fs =16000
            fake_IR = np.array(fake[i].to("cpu").detach())
            f = WaveWriter(fake_RIR_path, channels=1, samplerate=fs)
            f.write(np.array(fake_IR))
            
            # write(fake_RIR_path,fs,fake[i].astype(np.float32))

        # vutils.save_image(
        #     fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
        #     (RIR_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pth' % (model_dir))
    #print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
