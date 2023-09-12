import json
import os
from nltk.stem import PorterStemmer
from db_functions import create_json, load_json, dump_json, get_json_dict

# def create_json(name):
#     if not os.path.exists(name):
#         empty = {}
#         # Dump the updated parent_index to JSON file
#         with open(name, 'w') as f:
#             json.dump(empty, f)

# def load_json(json_name):
#     if os.path.exists(json_name):
#         with open(json_name) as f:
#             data = json.load(f)
#             return data

# def dump_json(json_name, data):
#     with open(json_name, 'w') as f:
#         json.dump(data, f)
#     return 0

def indexer():
    ps = PorterStemmer()
    with open('stopwords.txt', 'r') as f:
        stopwords = set([word.strip() for word in f])

     ##Create necessary json file if not exist
    if not os.path.exists("./db"):
        os.mkdir("./db")
    json_dict = get_json_dict()

    for name in json_dict:
        create_json(json_dict[name])

    raw_content_forward_index = load_json(json_dict['raw_content_forward_index'])
    raw_title_forward_index = load_json(json_dict['raw_title_forward_index'])
    content_inverted_index = load_json(json_dict['content_inverted_index'])
    content_forward_index = load_json(json_dict['content_forward_index'])
    title_inverted_index = load_json(json_dict['title_inverted_index'])
    title_forward_index = load_json(json_dict['title_forward_index'])
    wordid_to_word = load_json(json_dict['wordid_to_word'])
    stem_to_stemid = load_json(json_dict['stem_to_stemid'])
    stemid_to_stem = load_json(json_dict['stemid_to_stem'])

    max_stem_id = str(0)
    for page_id in raw_content_forward_index:
        content_forward_index[page_id] = []
        for i, word_id in enumerate(raw_content_forward_index[page_id]):
            pos = str(i)
            word = wordid_to_word[str(word_id)]
            if word in stopwords:
                continue
            stems = ps.stem(word)
            if stems not in stem_to_stemid:
                stemid_to_stem[max_stem_id] = stems
                stem_to_stemid[stems] = max_stem_id
                stem_id = max_stem_id
                max_stem_id = str(int(max_stem_id) + 1)
            else:
                stem_id = stem_to_stemid[stems]
            content_forward_index[page_id].append(stem_id)
            if stem_id not in content_inverted_index:
                content_inverted_index[stem_id] = {}
                content_inverted_index[stem_id][page_id] = [pos]
            elif page_id not in content_inverted_index[stem_id]:
                content_inverted_index[stem_id][page_id] = [pos]
            else:
                content_inverted_index[stem_id][page_id].append(pos)

    for page_id in raw_title_forward_index:
        title_forward_index[page_id] = []
        for i, word_id in enumerate(raw_title_forward_index[page_id]):
            pos = str(i)
            word = wordid_to_word[str(word_id)]
            if word in stopwords:
                continue
            stems = ps.stem(word)
            if stems not in stem_to_stemid:
                stemid_to_stem[max_stem_id] = stems
                stem_to_stemid[stems] = max_stem_id
                stem_id = max_stem_id
                max_stem_id = str(int(max_stem_id) + 1)
            else:
                stem_id = stem_to_stemid[stems]
            title_forward_index[page_id].append(stem_id)
            if stem_id not in title_inverted_index:
                title_inverted_index[stem_id] = {}
                title_inverted_index[stem_id][page_id] = [pos]
            elif page_id not in title_inverted_index[stem_id]:
                title_inverted_index[stem_id][page_id] = [pos]
            else:
                title_inverted_index[stem_id][page_id].append(pos)

    dump_json(json_dict['content_inverted_index'], content_inverted_index)
    dump_json(json_dict['content_forward_index'], content_forward_index)
    dump_json(json_dict['title_inverted_index'], title_inverted_index)
    dump_json(json_dict['title_forward_index'], title_forward_index)
    dump_json(json_dict['stem_to_stemid'], stem_to_stemid)
    dump_json(json_dict['stemid_to_stem'], stemid_to_stem)
    return 0
