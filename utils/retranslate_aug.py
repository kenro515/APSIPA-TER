import os
import glob
import time
from googletrans import Translator


if __name__ == '__main__':
    in_dir = "***"
    out_dir = "***"

    paths = sorted(glob.glob('{}/JTES/text/**/**.txt'.format(in_dir)))

    list_country_code = [
        'zh-CN', 'en', 'hi', 'es', 'ar', 'bn', 'pt', 'ru', 'de', 'fr', 'pa', 'jw', 'ko', 'vi'
    ]

    for country_code in list_country_code:
        basename = '{}/JTES_text/text_translated_augment/ja2{}2ja'.format(
            out_dir, country_code)
        os.makedirs(basename, exist_ok=True)

        os.makedirs(basename + '/ang', exist_ok=True)
        os.makedirs(basename + '/sad', exist_ok=True)
        os.makedirs(basename + '/joy', exist_ok=True)
        os.makedirs(basename + '/neu', exist_ok=True)

        for path in paths:
            f_r = open(path, 'r')
            lines = f_r.readlines()
            f_r.close()

            translator = Translator()

            for line in lines:
                translated = translator.translate(line, dest=country_code)
                time.sleep(1.0)
                trans_text = translated.text
                translated = translator.translate(trans_text, dest="ja")
                time.sleep(1.0)

                print('src_text:{}'.format(line))
                print('trg_text:{}'.format(trans_text))
                print('export_text:{}'.format(translated.text))

            emo_label = path.split('/')[7].split('_')[0]
            f_w = open(basename + '/' + emo_label +
                       '/' + path.split('/')[7], 'w')
            f_w.write(translated.text)
            f_w.close()

    print("finished!")
