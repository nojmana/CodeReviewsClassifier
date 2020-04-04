from Classifier import input_reader


if __name__ == "__main__":
    gerrit_file = 'gerrit-wireshark-train-test-v4.xlsx'
    data = input_reader.read_excel(gerrit_file, 'train')
