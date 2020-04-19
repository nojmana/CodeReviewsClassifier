import random

import pandas as pd
import xlrd
import csv
from sklearn import preprocessing


def read_csv(file_name):
    data = pd.read_csv(file_name)
    return data


def read_excel(file_name, sheet_name):
    data = pd.read_excel(file_name, sheet_name=sheet_name, index_col=None, usecols=['message', 'purpose'])
    return data.to_dict(orient='record')


def xlsx_to_csv(xlsx_filename):
    xlsx_file = xlrd.open_workbook(xlsx_filename)
    sheet = xlsx_file.sheet_by_name('Sheet1')
    csv_name = xlsx_filename.replace('xslx', 'csv')
    csv_file = open(csv_name, 'w')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

    for row_num in range(sheet.nrows):
        wr.writerow(sheet.row_values(row_num))
    csv_file.close()

    return csv_name


def split_set(data_set, index, seed):
    random.seed(seed)
    random.shuffle(data_set)

    split_index = int(index*len(data_set))
    return data_set[:split_index], data_set[split_index:]


def encode_labels(y):
    return preprocessing.LabelEncoder().fit_transform(y)
