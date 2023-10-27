import os
import numpy as np
import argparse
from multiprocessing import Pool
import random
import soundfile as sf
import scipy.signal as ssi
from tqdm import tqdm
import utility

def augment_data(speech_path, output_path, irfile_path, noise_path, SNR):

    speech, fs_s = sf.read(speech_path)
    if len(speech.shape) != 1:
        speech = speech[:, 0]
    if np.issubdtype(speech.dtype, np.integer):
        speech = utility.pcm2float(speech, 'float32')    
    # convolution
    if irfile_path:
        IR, fs_i = sf.read(irfile_path)
        if len(IR.shape) != 1:
            IR = IR[:, 0]
        if np.issubdtype(IR.dtype, np.integer):
            IR = utility.pcm2float(IR, 'float32')
        speech = utility.convert_samplerate(speech, fs_s, fs_i)
        fs_s = fs_i
        # eliminate delays due to direct path propagation
        direct_idx = np.argmax(np.fabs(IR))
        #print('speech {} direct index is {} of total {} samples'.format(speech_path, direct_idx, len(IR)))
        temp = utility.smart_convolve(speech, IR[direct_idx:])
        speech = np.array(temp)
    # adding noises
    if noise_path:
        noise, fs_n = sf.read(noise_path)
        if len(noise.shape) != 1:
            print("noise file should be single channel")
            return -1
        if np.issubdtype(noise.dtype, np.integer):
            noise = utility.pcm2float(noise, 'float32')
        noise = utility.convert_samplerate(noise, fs_n, fs_s)
        fs_n = fs_s       
        speech_len = len(speech)
        noise_len = len(noise)
        nrep = int(speech_len * 2 / noise_len)
        if nrep >= 1:
            noise = np.repeat(noise, nrep + 1)
            noise_len = len(noise)
        start = np.random.randint(noise_len - speech_len)
        noise = noise[start:(start + speech_len)]

        signal_power = utility.calc_valid_power(speech)
        noise_power = utility.calc_valid_power(noise)
        K = (signal_power / noise_power) * np.power(10, -SNR / 10)

        new_noise = np.sqrt(K) * noise
        speech = speech + new_noise
    maxval = np.max(np.fabs(speech))
    if maxval == 0:
        print("file {} not saved due to zero strength".format(speech_path))
        return -1
    if maxval >= 1:
        amp_ratio = 0.99 / maxval
        speech = speech * amp_ratio
    sf.write(output_path, speech, fs_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='augment',
                                     description="""Script to augment dataset""")
    parser.add_argument("--ir", "-i", default=None, help="Directory of IR files", type=str)
    parser.add_argument("--noise", "-no", default=None, help="Directory of noise files", type=str)
    parser.add_argument("--speech", "-sp", required=True, help="Directory of speech files", type=str)
    parser.add_argument("--out", "-o", required=True, help="Output folder path", type=str)
    parser.add_argument("--seed", "-s", default=0, help="Random seed", type=int)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

    args = parser.parse_args()
    speech_folder = args.speech
    noise_folder = args.noise
    ir_folder = args.ir
    output_folder = args.out
    nthreads = args.nthreads
    
    # force input and output folder to have the same ending format (i.e., w/ or w/o slash)
    speech_folder = os.path.join(speech_folder, '')
    output_folder = os.path.join(output_folder, '')    

    add_reverb = True if ir_folder else False
    add_noise = True if noise_folder else False

    assert os.path.exists(speech_folder)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    if add_noise:
        assert os.path.exists(noise_folder)
    if add_reverb:
        assert os.path.exists(ir_folder)

    speechlist = [os.path.join(root, name) for root, dirs, files in os.walk(speech_folder)
              for name in files if name.endswith(".wav")]
    irlist = [os.path.join(root, name) for root, dirs, files in os.walk(ir_folder)
              for name in files if name.endswith(".wav")] if add_reverb else []
    noiselist = [os.path.join(root, name) for root, dirs, files in os.walk(noise_folder)
              for name in files if name.endswith(".wav")] if add_noise else []

    # apply_async callback
    pbar = tqdm(total=len(speechlist))
    def update(*a):
        pbar.update()
    try:
        # # Create a pool to communicate with the worker threads
        pool = Pool(processes=nthreads)
        for speech_path in speechlist:
            ir_sample = random.choice(irlist) if add_reverb else None
            noise_sample = random.choice(noiselist) if add_noise else None
            SNR = np.random.uniform(10, 20)
            output_path = speech_path.replace(speech_folder, output_folder)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            pool.apply_async(augment_data, args=(speech_path, output_path, ir_sample, noise_sample, SNR), callback=update)
    except Exception as e:
        print(str(e))
        pool.close()
    pool.close()
    pool.join()

