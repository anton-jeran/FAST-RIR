import os, fnmatch
import numpy as np
import random
import soundfile as sf
from scipy.io.wavfile import write
# import librosa
import RT60

folder_path = "/cephfs/anton/room-impulse-responses/AIR/RWCP_REVERB_AACHEN/real_rirs_isotropic_noises/"
final_path = "/cephfs/anton/room-impulse-responses/AIR/RWCP_REVERB_AACHEN/AACHEN/"
tfs =16000
file_label = open("RT60.txt","w")

for root, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if filename.endswith(".wav"):
            ACE_Path = os.path.join(root, filename)
            wave,fs = sf.read(ACE_Path)
            channel = int(wave.size/len(wave))

            if(channel == 1):            
                wave_single = wave #librosa.resample(wave, fs, tfs)
                max_loc = np.where(wave_single == np.amax(wave_single))
                min_loc = np.where(wave_single == np.amin(wave_single))
                start = min(max_loc[0][0],min_loc[0][0])
                wave_single =wave_single[start:len(wave_single)]
                T60_val = RT60.t60_impulse(wave_single,tfs)
                
                if(T60_val<1):
                    file_label.write(str(T60_val)+"\n")
                    save_path = final_path+ filename
                    write(save_path,tfs,wave_single.astype(np.float32))
            else:
                for n in range(channel):
                    wave_single   = wave[:,n]#librosa.resample(wave[:,n], fs, tfs)
                    max_loc = np.where(wave_single == np.amax(wave_single))
                    min_loc = np.where(wave_single == np.amin(wave_single))
                    start = min(max_loc[0][0],min_loc[0][0])
                    wave_single =wave_single[start:len(wave_single)]
                    T60_val = RT60.t60_impulse(wave_single,tfs)
                    
                if(T60_val<1):
                        file_label.write(str(T60_val)+"\n")
                        save_path = final_path+filename+str(n)+".wav"
                        write(save_path,tfs,wave_single.astype(np.float32))
                    