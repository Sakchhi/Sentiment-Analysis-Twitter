"""
    Credits: https://github.com/rishabhverma17
"""

import csv
import re


def translator(text):
    text = text.split(" ")
    j = 0
    for _str in text:
        # File path which consists of Abbreviations.
        file_name = "slang_translator.txt"

        # File Access mode [Read Mode]
        access_mode = "r"
        with open(file_name, access_mode) as f:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            data = csv.reader(f, delimiter="=")

            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9]', '', _str)
            for row in data:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    text[j] = row[1]
            f.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(text)


if __name__ == '__main__':
    print(translator("brb. luv u"))
