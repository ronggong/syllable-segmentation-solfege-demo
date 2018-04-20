import subprocess


def synthesize_singing_mbrola(mbrola_path, soundfont, input_phn, output_wav):
    subprocess.call([mbrola_path, soundfont, input_phn, output_wav])


if __name__ == '__main__':
    synthesize_singing_mbrola(mbrola_path='./Mbrola/mbrola',
                              soundfont='./Mbrola/tr1/tr1',
                              input_phn='./Mbrola/solfege_demo.pho',
                              output_wav='./Mbrola/solfege_demo.wav')