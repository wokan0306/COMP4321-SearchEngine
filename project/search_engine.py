from nltk.stem import PorterStemmer
import numpy as np
import re
from collections import Counter
import math
from display import display_results
from db_functions import dump_json, create_json, load_json, get_json_dict
from HITS import hits_algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
        

def parse_query(query_terms, stem_to_stemid):
    """Parse the query into a {term: weight} dict"""
    ps = PorterStemmer()
    with open('stopwords.txt', 'r') as f:
        stopwords = set([word.strip() for word in f])
    query_terms = query_terms.lower()
    # Split query into terms
    pattern = r'\".+?\"|\'.+?\'|\w+'

    matches = re.findall(pattern, query_terms)
    stemmed_terms = []
    for word in matches:
        if " " in word:
            word = word.replace('\'',"")
            word = word.replace('\"',"")
            word = word.split(" ")
            word = [x for x in word if x not in stopwords]
            word = [ps.stem(x) for x in word]
            word = [stem_to_stemid[x] for x in word if x in stem_to_stemid]
            stemmed_terms.append(word)
            continue
        if word in stopwords:
            continue
        word = ps.stem(word)
        if word in stem_to_stemid:
            word = stem_to_stemid[word]
            stemmed_terms.append(word)

    return stemmed_terms

def count_successive_occurrences(term_dict):
    counter = {}
    
    # Get the set of document IDs present in the term dictionary
    doc_ids = set()
    for term_positions in term_dict.values():
        for doc_id in term_positions:
            doc_ids.add(doc_id)

    # Iterate through the document IDs
    for doc_id in doc_ids:
        count = 0

        # Get the positions for each term in the current document
        term_positions_list = [set(map(int, term_dict[term_id].get(doc_id, []))) for term_id in term_dict]

        # Iterate through the positions of the first term
        for pos in term_positions_list[0]:
            # Check for successive occurrences
            successive = True
            for i in range(1, len(term_positions_list)):
                if pos + i not in term_positions_list[i]:
                    successive = False
                    break

            if successive:
                count += 1

        if count > 0:
            counter[doc_id] = count

    return counter

def phrase_search_count(term_ids, inverted_index):
    # Return early if any term is missing from the index
    if not all(term_id in inverted_index for term_id in term_ids):
        return Counter()

    # Get the postings lists for all terms 
    postings = {term_id: inverted_index[term_id] for term_id in term_ids}

    results = count_successive_occurrences(postings)

    return results

def update_target_dict_phrase_search(order, item_length, score_dict, term_id, inverted_index, forward_index, query_scores, query_counter):
    PHRASE_BOOST = 3.0
    NUM_DOCS = len(forward_index) + 1 ## add 1 for the query
    matched_docs = phrase_search_count(term_id, inverted_index)
    if matched_docs == Counter():
        return query_scores, score_dict
    num_docs_with_phrase = len(matched_docs) + 1 ## add 1 for the query
    idf = math.log(NUM_DOCS / num_docs_with_phrase)
    query_tf = query_counter[term_id]
    for doc_id in matched_docs.keys():
        if doc_id not in score_dict:
            score_dict[doc_id] = np.zeros(item_length, dtype=np.float32)
        # Calculate tf, max_tf, and idf for the phrase
        tf = matched_docs[doc_id]
        max_tf = max(max(matched_docs.values(), default=0), query_tf)
        # Calculate tf-idf score
        score = tf * idf / max_tf * PHRASE_BOOST
        # Add to scores dict
        score_dict[doc_id][order] = score
    
    # Calculate tf and idf for the query term
    # Calculate tf-idf score
    score = query_tf * idf / max_tf
    query_scores[order] = score

    return query_scores, score_dict

def update_target_dict(order, item_length, score_dict, term_id, inverted_index, forward_index, query_scores, query_counter):
    
    
    if (term_id not in inverted_index):
        return query_scores, score_dict
    
    NUM_DOCS = len(forward_index) + 1 ## add 1 for the query
    num_docs_with_phrase = len(inverted_index[term_id])+1 ## add 1 for the query
    idf = np.log(NUM_DOCS / num_docs_with_phrase)  # Inverse document frequency

    max_tf = 0

    for doc_id, positions in inverted_index[term_id].items():
        tf = len(positions)
        if tf > max_tf:
            max_tf = tf
            
    if max_tf == 0:
        return query_scores, score_dict
    
    query_tf = query_counter[term_id]
    max_tf = max(query_tf, max_tf)

    # Add scores for documents containing this term
    for doc_id, positions in inverted_index[term_id].items():

        if doc_id not in score_dict:
            score_dict[doc_id] = np.zeros(item_length, dtype=np.float32)

        # Calculate tf-idf score
        tf = len(positions) # Term frequency
        score = tf * idf / max_tf

        # Add to scores dict
        score_dict[doc_id][order] = score
    
    # Calculate tf, max_tf, and idf for the query term
    NUM_DOCS = len(forward_index) + 1
    
    # Calculate tf-idf score
    score = query_tf * idf / max_tf
    query_scores[order] = score

    return query_scores, score_dict

def get_cosine_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)

    if query_norm == 0 or doc_norm == 0:
        return 0

    return dot_product / (query_norm * doc_norm)

def calculate_similarity(query_vector, score_dict):
    scores = {}
    for doc_id in score_dict.keys():
        scores[doc_id] = get_cosine_similarity(query_vector, score_dict[doc_id])
    return scores

def merge_dicts(A, B, ratio):
    merged = {
        key: (
            ratio * A[key] + (1-ratio) * B[key] if key in A and key in B
            else ratio * A.get(key, 0) + (1-ratio) * B.get(key, 0)
        )
        for key in set(A) | set(B)
    }
    return merged

def search(query_terms, content_inverted_index, content_forward_index, title_inverted_index,
                     title_forward_index, stem_to_stemid, child_index, query_dict=None):
    """Return top 50 results for the query"""
    
    json_dict = get_json_dict()

    PHRASE_BOOST = 3.0
    

    # Convert query to stem list and create a phrase array
    query_list = parse_query(query_terms, stem_to_stemid)

    # Create a target dictionary contain the position and mapping of each query term and document in terms of title and page body
    content_target_dict = {}
    title_target_dict = {}
    query_counter = Counter()
    for x in query_list:
        if type(x) == list:
            x = tuple(x)
        if x not in query_counter:
            query_counter[x] = 0
        query_counter[x] += 1
    query_unique_list = list(query_counter.keys())
    item_length = len(query_counter)
    content_query_scores = np.zeros(item_length, dtype=np.float32)
    title_query_scores = np.zeros(item_length, dtype=np.float32)
    
    # Iterate over each term in the query
    for i, term_id in enumerate(query_unique_list):
        if type(term_id) == str:
            content_query_scores, content_target_dict = update_target_dict(i, item_length, content_target_dict, term_id, content_inverted_index, content_forward_index, content_query_scores, query_counter)
            title_query_scores, title_target_dict = update_target_dict(i, item_length, title_target_dict, term_id, title_inverted_index, title_forward_index, title_query_scores, query_counter)
        else:
            content_query_scores, content_target_dict = update_target_dict_phrase_search(i, item_length, content_target_dict, term_id, content_inverted_index, content_forward_index, content_query_scores, query_counter)
            title_query_scores, title_target_dict = update_target_dict_phrase_search(i, item_length, title_target_dict, term_id, title_inverted_index, title_forward_index, title_query_scores, query_counter)
    
    # Calculate scores for each query term with each document in terms of title and page body
    content_similarity = calculate_similarity(content_query_scores, content_target_dict)
    title_similarity = calculate_similarity(title_query_scores, title_target_dict)
    similarity = merge_dicts(content_similarity, title_similarity, 0.25)

    # Sort the dictionary based on values in descending order
    sorted_items = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    
    if query_dict is not None:
        sorted_items = hits_algorithm(dict(sorted_items), child_index)

    # Get the top 50 items with the largest values
    top_50_items = sorted_items[:50]

    # Create a new dictionary with the top 50 items
    top_50_dict = dict(top_50_items)
    
    if query_dict is not None:
        query_dict[query_terms] = top_50_dict
        dump_json(json_dict['query'], query_dict)
        
    return top_50_dict

def get_most_similar_historical_query(current_query, historical_queries):
    vectorizer = TfidfVectorizer()
    
    # Concatenate the current query with the historical queries
    queries = [current_query] + historical_queries
    query_vectors = vectorizer.fit_transform(queries)
    
    # Calculate the cosine similarity between the current query and historical queries
    similarity_matrix = cosine_similarity(query_vectors)
    current_query_similarity = similarity_matrix[0, 1:]

    # Find the historical query with the highest similarity
    most_similar_query_index = current_query_similarity.argmax()
    most_similar_query = historical_queries[most_similar_query_index]

    return most_similar_query

def match_with_other_similarity(query_terms, doc_id, content_inverted_index, content_forward_index, title_inverted_index,
                     title_forward_index, stem_to_stemid, child_index, stemid_to_stem, page_info, pageid_to_url):
    
    counter = Counter(content_forward_index[doc_id])
    top_5 = counter.most_common(5)
    top_5 = [item[0] for item in top_5]
    content_query_terms = [stemid_to_stem[stem_id] for stem_id in top_5]
    content_query = " ".join(content_query_terms)
    content_query = query_terms + " " + content_query
    
    content_score_dict = search(content_query, content_inverted_index, content_forward_index, title_inverted_index,
                     title_forward_index, stem_to_stemid, child_index)
    if title_forward_index[doc_id]:
        title_query_terms = [stemid_to_stem[stem_id] for stem_id in title_forward_index[doc_id]]
        title_query =  " ".join(title_query_terms)
        title_score_dict = search(title_query, content_inverted_index, content_forward_index, title_inverted_index,
                     title_forward_index, stem_to_stemid, child_index)
        similarity = merge_dicts(content_score_dict, title_score_dict, 0.25)
        # Sort the dictionary based on values in descending order
        sorted_items = dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[1:6])
    else:
        # Sort the dictionary based on values in descending order
        sorted_items = dict(sorted(content_score_dict.items(), key=lambda x: x[1], reverse=True)[1:6])
    output = [(page_info[doc_id]["Page title"], pageid_to_url[doc_id]) for doc_id in sorted_items]
    return output
    
def search_engine(query_terms, page_info, stemid_to_stem, pageid_to_url, content_inverted_index,
                  content_forward_index, title_inverted_index, title_forward_index, stem_to_stemid, child_index, query_dict):

    outputs = search(query_terms, content_inverted_index = content_inverted_index, content_forward_index= content_forward_index, title_inverted_index=title_inverted_index,
                     title_forward_index=title_forward_index, stem_to_stemid=stem_to_stemid, child_index=child_index, query_dict=query_dict)
    if query_dict:
        best_match_hist_item = get_most_similar_historical_query(query_terms, list(query_dict.keys()))
        suboptimal_hist_scores = query_dict[best_match_hist_item]
        outputs = merge_dicts(suboptimal_hist_scores, outputs, 0.05)
        outputs = dict(sorted(outputs.items(), key=lambda x: x[1], reverse=True))
    other_results = {}
    for doc_id in outputs:
        other_results[doc_id] = match_with_other_similarity(query_terms, doc_id, content_inverted_index, content_forward_index, title_inverted_index,
                     title_forward_index, stem_to_stemid, child_index, stemid_to_stem, page_info, pageid_to_url)
    for key in outputs:
        outputs[key] = outputs[key] * 100
    query_results = display_results(outputs)
    return query_results, other_results

    
if __name__ == '__main__':
    query_terms = '\"Hong Kong\" Book Essay'
    top_50_dict = search_engine(query_terms)
    print(top_50_dict)