import io
import glob

if __name__ == '__main__':
    in_dir = "***"
    out_dir = "***"

    paths_train = sorted(
        glob.glob("{}/**/*.txt".format(in_dir)))
    fw = open(out_dir, 'w')

    for path in paths_train:
        with io.open(path, 'r', encoding="utf-8") as fr:
            text = fr.readline()
            text = text.replace('\t', " ")
            text = text.replace('\n', " ")

            emo_label = path.split('/')[6]
            if emo_label == 'ang':
                text = text + '\t' + '0' + '\t' + '\n'
            elif emo_label == 'joy':
                text = text + '\t' + '1' + '\t' + '\n'
            elif emo_label == 'sad':
                text = text + '\t' + '2' + '\t' + '\n'
            else:
                text = text + '\t' + '3' + '\t' + '\n'

            fw.write(text)
    fw.close()
