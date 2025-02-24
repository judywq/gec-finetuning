from enum import Enum
import os
import pandas as pd
import json


def save_to_jsonl(dataset, file_path):
    path = os.path.dirname(file_path)
    os.makedirs(path, exist_ok=True)
    with open(file_path, 'w') as file:
        for record in dataset:
            json_line = json.dumps(record)
            file.write(json_line + '\n')

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_to_json(data: dict, file_path, indent=4):
    if 'error' in data:
        del data['error']
    if 'method' in data:
        del data['method']
    path = os.path.dirname(file_path)
    os.makedirs(path, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=indent)


def read_data(path) -> pd.DataFrame:
    df = None
    _type = parse_file_type(path)
    if _type == FileType.CSV:
        df = pd.read_csv(path)
    elif _type == FileType.EXCEL:
        df = pd.read_excel(path)
    return df


def write_data(df: pd.DataFrame, filename: str, index=False):
    path = os.path.dirname(filename)
    if path:
        os.makedirs(path, exist_ok=True)
    _type = parse_file_type(filename)
    if _type == FileType.CSV:
        df.to_csv(filename, index=index)
    elif _type == FileType.EXCEL:
        df.to_excel(filename, index=index)


class FileType(Enum):
    CSV = 'csv'
    EXCEL = 'excel'
    
type_ext_map = {
    FileType.CSV: ['csv'],
    FileType.EXCEL: ['xls', 'xlsx'],
}

def parse_file_type(path):
    ext = (path.split('.')[-1]).lower()
    
    for _type, type_list in type_ext_map.items():
        if ext in type_list:
            return _type

    return None


################
# Test
################

def test_io():
    path = 'data/input/keywords.xlsx'
    df = read_data(path)
    print(df)
    
    out_path = 'data/output/test.xlsx'
    write_data(df, out_path)


if __name__ == '__main__':
    test_io()
    