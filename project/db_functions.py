import json
import os
from nltk.tokenize import RegexpTokenizer

def create_json(name):
    if not os.path.exists(name):
        empty = {}
        # Dump the updated parent_index to JSON file
        with open(name, 'w') as f:
            json.dump(empty, f)

def load_json(json_name):
    if os.path.exists(json_name):
        with open(json_name) as f:
            data = json.load(f)
            return data

def dump_json(json_name, data):
    with open(json_name, 'w') as f:
        json.dump(data, f)
    return 0

def get_json_dict():
    return {
    'child_index': './db/child_index.json',
    'parent_index': './db/parent_index.json',
    'raw_content_forward_index': './db/raw_content_forward_index.json',
    'raw_title_forward_index': './db/raw_title_forward_index.json',
    'content_forward_index': './db/content_forward_index.json',
    'content_inverted_index': './db/content_inverted_index.json',
    'title_forward_index': './db/title_forward_index.json',
    'title_inverted_index': './db/title_inverted_index.json',
    'url_to_pageid': './db/url_to_pageid.json',
    'pageid_to_url': './db/pageid_to_url.json',
    'wordid_to_word': './db/wordid_to_word.json',
    'stem_to_stemid': './db/stem_to_stemid.json',
    'stemid_to_stem': './db/stemid_to_stem.json',
    'page_info': './db/page_info.json',
    "query": './db/query.json'
    }

def word_register(page_id, word_list, word_to_wordid, wordid_to_word, inverted_index, forward_index):
    for i, word in enumerate(word_list):
        if word not in word_to_wordid:
            if not word_to_wordid:
                word_id = 1
            else:
                word_id = max(int(key) for key in wordid_to_word) + 1
            word_to_wordid[word] = word_id
            wordid_to_word[word_id] = word
        else:
            word_id = word_to_wordid[word]
        if word_id in inverted_index:
            if page_id not in inverted_index[word_id]:
                inverted_index[word_id][page_id] = [i]
            else:
                inverted_index[word_id][page_id].append(i)
        else:
            inverted_index[word_id] = {page_id:[i]}
        forward_index[page_id].append(word_id)

    return word_to_wordid, wordid_to_word, inverted_index, forward_index

def process_item(item):

    ##Create necessary json file if not exist
    if not os.path.exists("./db"):
        os.mkdir("./db")
    json_dict = {
    'child_index': './db/child_index.json',
    'parent_index': './db/parent_index.json',
    'raw_content_forward_index': './db/raw_content_forward_index.json',
    'raw_content_inverted_index': './db/raw_content_inverted_index.json',
    'raw_title_forward_index': './db/raw_title_forward_index.json',
    'raw_title_inverted_index': './db/raw_title_inverted_index.json',
    'url_to_pageid': './db/url_to_pageid.json',
    'pageid_to_url': './db/pageid_to_url.json',
    'word_to_wordid': './db/word_to_wordid.json',
    'wordid_to_word': './db/wordid_to_word.json',
    'page_info': './db/page_info.json'
    }

    for name in json_dict:
        create_json(json_dict[name])

    # Load the existing data dictionaries if files exist, else create empty ones
    page_info = load_json(json_dict['page_info'])
    url_to_pageid = load_json(json_dict['url_to_pageid'])
    pageid_to_url = load_json(json_dict['pageid_to_url'])
    child_index = load_json(json_dict['child_index'])
    parent_index = load_json(json_dict['parent_index'])
    raw_content_inverted_index = load_json(json_dict['raw_content_inverted_index'])
    raw_content_forward_index = load_json(json_dict['raw_content_forward_index'])
    raw_title_inverted_index = load_json(json_dict['raw_title_inverted_index'])
    raw_title_forward_index = load_json(json_dict['raw_title_forward_index'])
    word_to_wordid = load_json(json_dict['word_to_wordid'])
    wordid_to_word = load_json(json_dict['wordid_to_word'])

    url = item['URL']
    if not url_to_pageid:
        page_id = 1
        url_to_pageid[url] = page_id
        pageid_to_url[page_id] = url
    else:
        if url in url_to_pageid:
            page_id = url_to_pageid[url]
        else:
            page_id = max([int(key) for key in pageid_to_url]) + 1
            url_to_pageid[url] = page_id
            pageid_to_url[page_id] = url

    # Add item to the data dictionaries
    word_list = item['Words']

    raw_content_forward_index[page_id] = []
    raw_title_forward_index[page_id] = []

    # Remove stopwords from tokens
    word_list_content = [token.lower() for token in word_list]
    tokenizer = RegexpTokenizer(r'[a-z]+')
    title_tokens = tokenizer.tokenize(item['Page title'])
    word_list_title = [token.lower() for token in title_tokens]
    word_to_wordid, wordid_to_word, raw_content_inverted_index, raw_content_forward_index = word_register(page_id, word_list_content, word_to_wordid, wordid_to_word, raw_content_inverted_index, raw_content_forward_index)
    word_to_wordid, wordid_to_word, raw_title_inverted_index, raw_title_forward_index = word_register(page_id, word_list_title, word_to_wordid, wordid_to_word, raw_title_inverted_index, raw_title_forward_index)

    child_links = item['Child links']
    child_page_id = max(int(key) for key in pageid_to_url) + 1
    for link in child_links:
        if link not in url_to_pageid:
            url_to_pageid[link] = child_page_id
            pageid_to_url[child_page_id] = link
            child_page_id += 1
    child_index[url_to_pageid[url]] = [url_to_pageid[link] for link in child_links]

    for link in child_links:
        if url_to_pageid[link] not in parent_index:
            parent_index[url_to_pageid[link]] = [url_to_pageid[url]]
        else:
            parent_index[url_to_pageid[link]].append(url_to_pageid[url])

    # Create info dict for page
    page_info[page_id] =  {
        'Page title': item['Page title'],
        'Last modification date': item['Last modification date'],
        'Size of page': item['Size of page']
    }

    # Dump the updated data dictionaries into JSON files
    dump_json(json_dict['child_index'], child_index)
    dump_json(json_dict['parent_index'], parent_index)
    dump_json(json_dict['raw_content_inverted_index'], raw_content_inverted_index)
    dump_json(json_dict['raw_content_forward_index'], raw_content_forward_index)
    dump_json(json_dict['raw_title_inverted_index'], raw_title_inverted_index)
    dump_json(json_dict['raw_title_forward_index'], raw_title_forward_index)
    dump_json(json_dict['url_to_pageid'], url_to_pageid)
    dump_json(json_dict['word_to_wordid'], word_to_wordid)
    dump_json(json_dict['page_info'], page_info)
    dump_json(json_dict['wordid_to_word'], wordid_to_word)
    dump_json(json_dict['pageid_to_url'], pageid_to_url)

    return 0
