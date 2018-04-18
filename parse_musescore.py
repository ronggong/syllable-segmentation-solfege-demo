import os
import music21


def midi_event_parser(fn):
    mf = music21.midi.MidiFile()
    mf.open(fn)
    mf.read()
    mf.close()

    s = music21.midi.translate.midiFileToStream(mf)

    part = s.parts[0]

    part_tuples = []
    for event in part:
        for y in event.contextSites():
            if y[0] is part:
                offset = y[1]
        if getattr(event, 'isNote', None) and event.isNote:
            part_tuples.append((event.name, event.quarterLength, offset, event.nameWithOctave, event.frequency))
        if getattr(event, 'isRest', None) and event.isRest:
            part_tuples.append(('Rest', event.quarterLength, offset))

    return part_tuples


def convert_event_tuples_2_duration_label(event_tuples):
    durs = []
    labels = []
    for e in event_tuples:
        if e[0] != 'Rest':
            durs.append(e[1])
            labels.append(e[0])
        else:
            durs[-1] += e[1]
    return durs, labels


def convert_note_name_to_pho(note_name):
    if note_name == 'C':
        out = ['d', 'O']
    elif note_name == 'D':
        out = ['r', 'E']
    elif note_name == 'E':
        out = ['m', 'i']
    elif note_name == 'F':
        out = ['f', 'A']
    elif note_name == 'G':
        out = ['s', 'O']
    elif note_name == 'A':
        out = ['l', 'A']
    elif note_name == 'B':
        out = ['s', 'i']
    else:
        raise ValueError('{} doesn''t exist.'.format(note_name))
    return out


def convert_event_2_mbrola_format(event_tuples, tempo):
    unit_length = 60.0*1000.0/tempo
    out_syl_list = []
    for e in event_tuples:
        dur = e[1] * unit_length
        if e[0] != 'Rest':
            consonant, vowal = convert_note_name_to_pho(e[0])
            list_syl = [[consonant, str(100), str(0), str(int(e[-1]))], [vowal, str(int(dur-100))]]
        else:
            list_syl = [['_', str(int(dur))]]

        out_syl_list += list_syl
    return out_syl_list


if __name__ == '__main__':

    filename_exercise = '/home/gong/PycharmProjects/syllable-segmentation-solfege-demo/Dannhauser-exercises/exercise-43.mid'
    part_tuples = midi_event_parser(filename_exercise)
    tempo = 88
    note_seq_demo = part_tuples[1:18]

    durs, labels = convert_event_tuples_2_duration_label(note_seq_demo)
    out_syl_list = convert_event_2_mbrola_format(note_seq_demo, tempo)

    with open(os.path.join('./Mbrola/', 'solfege_demo.pho'), 'w') as textfile:
        for syl_list in out_syl_list:
            if len(syl_list) == 2:
                textfile.write("{} {}\n".format(syl_list[0], syl_list[1]))
            else:
                textfile.write("{} {} {} {}\n".format(syl_list[0], syl_list[1], syl_list[2], syl_list[3]))

    with open(os.path.join('.', 'solfege_score.txt'), 'w') as textfile:
        for l, d in zip(labels, durs):
            if len(syl_list) == 2:
                textfile.write("{} {}\n".format(l, d))