# Author: haihua
# 2020-05-08
# Extract case fulltext from json file and split the text into single sentences

import json
import tools.lexpredict.lexnlp.nlp.en.segments.sentences as lexSenSplit
from xml.dom.minidom import parseString


def getCaseLawData(inputFile):

    # the data was save as json file, parse the meta information and text from the json file
    data = [json.loads(line) for line in open(inputFile, 'r')]
    print(len(data))
    for case in data:
        all_sentences = []
        # print(case)
        caseId = case['id']
        print(caseId)
        docket_number = case['docket_number']
        print(docket_number)
        caseBody_text = case['casebody']['data']
        # print(caseBody_text)
        dom3 = parseString(caseBody_text)
        itemlist_p = dom3.getElementsByTagName('p')
        if len(itemlist_p)>0:
            for item in itemlist_p:
                paragraph = item.firstChild.nodeValue
                # print(paragraph)
                if paragraph != None:
                    sents = lexSenSplit.get_sentence_list(paragraph)
                    # print(sents)
                    for sent in sents:
                        all_sentences.append(sent)

        itemlist_blockquote = dom3.getElementsByTagName('blockquote')
        if len(itemlist_blockquote)>0:
            for item in itemlist_blockquote:
                blockquote = item.firstChild.nodeValue
                # print(blockquote)
                if blockquote !=None:
                    sents = lexSenSplit.get_sentence_list(blockquote)
                    # print(sents)
                    for sent in sents:
                        all_sentences.append(sent)

        print(all_sentences)
        filename = str(caseId) + '_' + str(docket_number)
        with open('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/HarvardCaseLawSentences' + '/' + filename, 'w') as outfile:
            for setnece in all_sentences:
                outfile.write('%s\n' % setnece)


if __name__ == '__main__':
    inputFile = '/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/HarvardCaseLaw/data.jsonl'
    getCaseLawData(inputFile)


