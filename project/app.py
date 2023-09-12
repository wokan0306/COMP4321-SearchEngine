from flask import Flask, render_template, request, send_from_directory
from search_engine import search_engine
from db_functions import dump_json, create_json, load_json
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/db/<path:filename>', methods=['GET'])
def serve_db_file(filename):
    return send_from_directory('db', filename)

@app.route('/search', methods=['POST'])
def search():
    json_dict = {
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
    page_info = load_json(json_dict["page_info"])
    stemid_to_stem= load_json(json_dict["stemid_to_stem"])
    pageid_to_url = load_json(json_dict["pageid_to_url"])
    
    content_inverted_index = load_json(json_dict['content_inverted_index'])
    content_forward_index = load_json(json_dict['content_forward_index'])
    title_inverted_index = load_json(json_dict['title_inverted_index'])
    title_forward_index = load_json(json_dict['title_forward_index'])
    stem_to_stemid = load_json(json_dict['stem_to_stemid'])
    child_index = load_json(json_dict['child_index'])
    query_dict = load_json(json_dict['query'])
    
    search_query = request.form.get('search')

    # Perform search and fetch results
    start_time = time.time()
    search_results, other_results = search_engine(search_query, page_info, stemid_to_stem, pageid_to_url, content_inverted_index,
                                                  content_forward_index, title_inverted_index, title_forward_index, stem_to_stemid, child_index, query_dict) # Replace this with your search function
    end_time = time.time()
    print(f"Retrieved results in {end_time-start_time} seconds")
    return render_template('search_results.html', search_query=search_query, search_results=search_results, other_results=other_results)

if __name__ == '__main__':
    app.run(debug=True)
