import os
import librosa
import utils
import numpy as np

Fs = 16000  # Hz
Ts = 1000.0/Fs  # ms

overlap = 10  # ms
sample_over = overlap/Ts

frame_time = 25  # ms
sample_frame = int(frame_time/Ts)

def calculate_average_number_frame(wave_files):
    print("\n#####Calculate average number of frame:")
    numFrameTotal = 0
    for wave_file in wave_files.keys():
        wave, _ = librosa.load(utils.DATA_PATH + wave_file, mono=True, sr=16000)
        nbFrame = int((len(wave) - sample_frame)/sample_over) + 1
        numFrameTotal += nbFrame
        print(wave_file + "\t: " + str(nbFrame))

    averNbFrame = int(numFrameTotal / len(wave_files))
    return averNbFrame

def recalculate_overlap(wave_files, averNbFrame):
    print("\n#####Recalculate overlap:")
    for wave_file in wave_files.keys():
        wave, _ = librosa.load(utils.DATA_PATH + wave_file, mono=True, sr=16000)
        sample_over_ = int((len(wave) - sample_frame)/(averNbFrame - 1))
        overlap_ = int(sample_over_ * Ts)
        temp_dict = dict()
        temp_dict[0] = sample_over_
        temp_dict[1] = overlap_
        wave_files[wave_file] = temp_dict
        print(wave_file + "\t:" + str(sample_over_) + "\t:" + str(overlap_))

def write_mfcc_f0_file(wave_files):
    for wave_file in wave_files.keys():
        temp_dict = wave_files[wave_file]
        sample_over_ = temp_dict[0]
        overlap_ = temp_dict[1]
        wave, _ = librosa.load(utils.DATA_PATH + wave_file, mono=True, sr=16000)
        print("\n#####Write to MFCC files: " + wave_file + "\t" + str(len(wave)) + "\t" + str(overlap_) + "\t" + str(sample_over_))
        f0 = np.loadtxt(utils.FO_PATH + wave_file[:wave_file.index(".")] + ".f0")
        with open(utils.MFCC_PATH + wave_file[:wave_file.index(".")] + ".txt", "wt") as f:
            for i in range(averNbFrame):
                start = i * sample_over_
                end = min(start + sample_frame, len(wave))
                wave_frame = wave[start:end]
                mfcc = librosa.feature.mfcc(
                    wave_frame, sr=16000, n_mfcc=averNbFrame)
                for j in range(len(mfcc)):
                    f.write(str(mfcc[j][0]))
                    # if (j < len(mfcc) - 1):
                    f.write("\t")
                f.write(str(f0[i]))
                if (i < averNbFrame - 1):
                    f.write("\n")
            print(str(i) + "\tstart=" + str(start) + "\tend=" + str(end))
        # break

if __name__ == '__main__':
    wave_files = dict()
    for wave_file in os.listdir(utils.DATA_PATH):
        wave_files[wave_file] = ""
    
    averNbFrame = calculate_average_number_frame(wave_files)
    print("averNbFrame = ", averNbFrame)

    recalculate_overlap(wave_files, averNbFrame)
    
    write_mfcc_f0_file(wave_files)