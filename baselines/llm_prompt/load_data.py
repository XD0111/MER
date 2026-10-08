# -*- coding: utf-8 -*-#
import json


# -------------------------------------------------------------------------------
# Name:         load_data
# Description:
# Author:       梁超
# Date:         2023/12/2
# -------------------------------------------------------------------------------

def load_jsonl(file_path):
    """
    Load data from a JSONL file.
    
    A JSONL file is a plain text file that contains one JSON object per line.
    This function reads the file and returns a list of dictionaries, where each
    dictionary corresponds to a JSON object read from a line in the file.
    
    Parameters:
    file_path (str): The path to the JSONL file.
    
    Returns:
    list: A list of dictionaries, each representing a JSON object from the file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Decode JSON object from each line and append to the list
            data.append(json.loads(line))
    return data


def load_json(file_path):
    """
    Load and parse the content of a JSON file from the specified path.

    Parameters:
    file_path: str, the path to the JSON file.

    Returns:
    A dictionary representing the parsed JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    valid_path = './new_data5/valid.json'
    valid_data = load_jsonl(valid_path)
    print(len(valid_data))
    test_path = './new_data5/test.json'
    test_data = load_jsonl(test_path)
    print(len(test_data))