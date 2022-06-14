import csv


def create_tsv_files(
    source_lang,
    target_lang,
    data_directory='data/iwslt/',
    train=False,
    dev=False,
    test=False
):
    train_filename = ['{source}-{target}/train.{source}-{target}.{lang}']
    dev_filename = ['{source}-{target}/IWSLT14.TED.dev2010.{source}-{target}.{lang}']
    if source_lang in ['en', 'de'] and target_lang in ['de', 'en']:
        dev_filename.append('{source}-{target}/IWSLT14.TEDX.dev2012.{source}-{target}.{lang}')
    test_filename = [
        '{source}-{target}/IWSLT14.TED.tst2010.{source}-{target}.{lang}',
        '{source}-{target}/IWSLT14.TED.tst2011.{source}-{target}.{lang}',
        '{source}-{target}/IWSLT14.TED.tst2012.{source}-{target}.{lang}'
    ]

    if train:
        final_train_filename = '{source}-{target}/train.{source}-{target}.tsv'.format(source=source_lang, target=target_lang)
        preprocess(final_train_filename, train_filename, source_lang, target_lang)
    if dev:
        final_dev_filename = '{source}-{target}/dev.{source}-{target}.tsv'.format(source=source_lang, target=target_lang)
        preprocess(final_dev_filename, dev_filename, source_lang, target_lang)
    if test:
        final_test_filename = '{source}-{target}/test.{source}-{target}.tsv'.format(source=source_lang, target=target_lang)
        preprocess(final_test_filename, test_filename, source_lang, target_lang)


def preprocess(output_file, input_files, source_lang, target_lang):
    train_data = []
    source_data = []
    target_data = []
    for file in input_files:
        source_file = file.format(source=source_lang, target=target_lang, lang=source_lang)
        target_file = file.format(source=source_lang, target=target_lang, lang=target_lang)

        with open(source_file, 'r', encoding='utf-8') as f:
            _data = [l.strip() for l in f]
            source_data.extend(_data)
        f.close()
        with open(target_file, 'r', encoding='utf-8') as f:
            _data = [l.strip() for l in f]
            target_data.extend(_data)
        f.close()

    assert len(source_data) == len(target_data)

    for i in range(len(source_data)):
        train_data.append([source_data[i], target_data[i]])

    with open(output_file, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerows(train_data)
    f.close()
