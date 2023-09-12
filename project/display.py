from db_functions import *
from collections import Counter

def display_results(score_dict):
    if not os.path.exists("./db"):
        return
    json_dict = {
    'child_index': './db/child_index.json',
    'parent_index': './db/parent_index.json',
    'forward_index': './db/content_forward_index.json',
    'inverted_index': './db/content_inverted_index.json',
    'url_to_pageid': './db/url_to_pageid.json',
    'pageid_to_url': './db/pageid_to_url.json',
    'stemid_to_stem': './db/stemid_to_stem.json',
    'page_info': './db/page_info.json'
    }
    # Load the existing data dictionaries if files exist, else create empty ones
    page_info = load_json(json_dict['page_info'])
    pageid_to_url = load_json(json_dict['pageid_to_url'])
    child_index = load_json(json_dict['child_index'])
    parent_index = load_json(json_dict['parent_index'])
    forward_index = load_json(json_dict['forward_index'])
    stemid_to_stem = load_json(json_dict['stemid_to_stem'])
    
    output = []

    if score_dict == None:
        return output
    for page_id in score_dict.keys():
        word_count = Counter(forward_index[page_id])
        word_count = sorted(word_count.items(), key=lambda item: (-item[1], item[0]))
        word_count = [(stemid_to_stem[word[0]],word[1]) for word in word_count]
        parent_links = [pageid_to_url[str(parent_id)] for parent_id in parent_index[page_id]]
        child_links = [pageid_to_url[str(child_id)] for child_id in child_index[page_id]]
        url = pageid_to_url[page_id]
        title = page_info[page_id]["Page title"]
        date = page_info[page_id]["Last modification date"]
        page_size = str(page_info[page_id]["Size of page"])
        page_information = {}
        page_information["score"] = "{:.4f}".format(score_dict[page_id])
        page_information["page_id"] = page_id
        page_information["title"] = title
        page_information["last_modified"] = date
        page_information["size"] = page_size
        page_information["parent_links"] = parent_links
        page_information["child_links"] = child_links
        page_information["url"] = url
        page_information["keywords"] = word_count
        output.append(page_information)
        
    return output

