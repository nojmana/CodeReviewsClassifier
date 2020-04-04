import pandas as pd
import xlrd
import csv


def read_csv(file_name):
    data = pd.read_csv(file_name)
    return data


def read_excel(file_name, sheet_name):
    data = pd.read_excel(file_name, sheet_name=sheet_name)
    return data


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
