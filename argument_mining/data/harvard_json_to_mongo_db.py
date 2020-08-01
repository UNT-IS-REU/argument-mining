# Author: haihua
# 2020-05-09
# Save the harvard case law data into mongoDB

# Standard Library Imports
import json
from json import JSONDecodeError
import os

# Third Party Library Imports
from ftfy import fix_text
from pymongo import MongoClient

# Local Library Imports


# For the regular json files
# def readJsonFile(inputFile):
#     # the data was save as json file, parse the meta information and text from the json file
#     data = [json.loads(line) for line in open(inputFile, 'r',encoding='utf-8')]
#     print(data)
#     return data

# For some files that are abnormal
def readJsonFile(inputFile):
    # the data was save as json file, parse the meta information and text from the json file
    # data = [json.loads(line) for line in open(inputFile, 'r',encoding='utf-8')]
    client = MongoClient()
    client = MongoClient("localhost", 27017)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['harvardCaseLaw']
    AlldataByJurisdiction = db.AlldataByJurisdiction
    data = []
    for line in open(inputFile, 'r',encoding='utf-8'):
        try:
            data.append(line)
            print(type(line))
            new_line = json.loads(line)
            records = AlldataByJurisdiction.insert_one(new_line)
        except JSONDecodeError:
            print("Unterminated string starting at")
            pass


def connectMongoDB(inputFile):
    client = MongoClient()
    client = MongoClient("localhost", 27017)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['harvardCaseLaw']
    AlldataByJurisdiction = db.AlldataByJurisdiction
    data = readJsonFile(inputFile)
    records = AlldataByJurisdiction.insert_many(data)
    print("The new records IDs are {}".format(records.inserted_ids))


if __name__ == '__main__':
    arr = os.listdir('/home/iialab/PycharmProjects/hardvardCaseLawDataCollection/')
    # print(arr)
    for item in arr:
        inputFile = '/home/iialab/PycharmProjects/hardvardCaseLawDataCollection/' + str (item)
        print(inputFile)
        readJsonFile(inputFile)
    # inputFile = '/home/iialab/PycharmProjects/hardvardCaseLawDataCollection/Pennsylvania.jsonl'
    # inputFile = '/home/iialab/PycharmProjects/hardvardCaseLawDataCollection/Delaware.jsonl'
    # readJsonFile(inputFile)
    # connectMongoDB(inputFile)
