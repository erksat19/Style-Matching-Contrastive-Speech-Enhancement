import os
import csv


def parse_metadata(metadata_dir):
    stt_dict = dict()
    with open(metadata_dir) as metadata_file:
        csv_reader = csv.reader(metadata_file)
        next(csv_reader) # omit first line
        for row in csv_reader: # ID, duration, wav, spk_id, wrd
            stt_dict[row[2]] = row[4]
    return stt_dict