"""
Run this script to create three files for the standard train, development, and test splits,
respectively using Section 02-21, 00, and 23.
Place this file under ccgbank_1_1/data/AUTO or edit the data_dir variable before running.
"""
import os

data_dir = '.'
split_to_sections = {
    'train':    ['02', '03', '04', '05', '06', '07', '08', '09', '10',
                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
    'dev':      ['00'],
    'test':     ['23'],
}

for split, sections in split_to_sections.items():
    output_file = f'ccgbank_{split}.auto'
    for section in sections:
        for file in sorted(os.listdir(os.path.join(data_dir, section))):
            with open(os.path.join(data_dir, output_file), "a+") as f:
                f.writelines([
                    line.strip()+'\n'
                    for line in open(os.path.join(data_dir, section, file), "r").readlines()
                ])

