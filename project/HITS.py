import numpy as np
from db_functions import load_json

def initialize_scores(pages):
    initial_score = 1 / len(pages)
    return {page: initial_score for page in pages}

def calculate_auth_hub_scores(adj_matrix, auth_scores, hub_scores, iterations=50):
    for _ in range(iterations):
        new_auth_scores = np.dot(adj_matrix.T, hub_scores)
        new_hub_scores = np.dot(adj_matrix, auth_scores)

        # Normalize the scores
        auth_scores = new_auth_scores / np.linalg.norm(new_auth_scores)
        hub_scores = new_hub_scores / np.linalg.norm(new_hub_scores)

    return auth_scores, hub_scores

import numpy as np

def build_transition_matrix(links, pages):
    n = len(pages)
    transition_matrix = np.zeros((n, n))
    page_index = {page: i for i, page in enumerate(pages)}

    for page, linked_pages in links.items():
        for linked_page in linked_pages:
            if linked_page in page_index:
                transition_matrix[page_index[page], page_index[linked_page]] = 1 / len(linked_pages)

    return transition_matrix

def pagerank(links, alpha=0.85, max_iter=100, tol=1e-6):
    pages = set(links.keys())
    for linked_pages in links.values():
        pages |= set(linked_pages)

    transition_matrix = build_transition_matrix(links, pages)
    n = len(pages)

    # Initialize the PageRank vector
    pagerank_vector = np.ones(n) / n

    # Power iteration
    for _ in range(max_iter):
        new_pagerank_vector = alpha * np.dot(transition_matrix, pagerank_vector) + (1 - alpha) / n

        if np.linalg.norm(new_pagerank_vector - pagerank_vector) < tol:
            break
        pagerank_vector = new_pagerank_vector

    return {page: pagerank_vector[i] for i, page in enumerate(pages)}


def combine_similarity_pagerank(similarity_scores, child_links, alpha=0.85, max_iter=100, tol=1e-6):
    # Calculate the PageRank scores
    pagerank_scores = pagerank(child_links, alpha, max_iter, tol)

    # Combine the similarity and PageRank scores
    combined_scores = {
        page: (similarity_scores.get(page, 0) + pagerank_scores.get(page, 0))
        for page in set(similarity_scores.keys()) | set(pagerank_scores.keys())
    }

    # Sort the pages by their combined scores
    ranked_pages = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_pages


def hits_algorithm(similarity_scores, child_links, iterations=50):
    pages = set(similarity_scores.keys()) | set(child_links.keys())
    for child_pages in child_links.values():
        pages |= set(child_pages)

    auth_scores = initialize_scores(pages)
    hub_scores = initialize_scores(pages)

    # Build the adjacency matrix
    adj_matrix = np.zeros((len(pages), len(pages)))
    page_index = {page: i for i, page in enumerate(pages)}

    for page, child_pages in child_links.items():
        for child_page in child_pages:
            if child_page in page_index:
                adj_matrix[page_index[page], page_index[child_page]] = 1

    # Convert dictionary auth_scores and hub_scores to numpy arrays
    auth_scores_array = np.array([auth_scores[page] for page in pages])
    hub_scores_array = np.array([hub_scores[page] for page in pages])

    auth_scores_array, hub_scores_array = calculate_auth_hub_scores(adj_matrix, auth_scores_array, hub_scores_array, iterations)

    # Convert the numpy arrays back to dictionaries
    auth_scores = {page: auth_scores_array[i] for i, page in enumerate(pages)}
    hub_scores = {page: hub_scores_array[i] for i, page in enumerate(pages)}

    # Combine the authority scores with the similarity scores
    final_scores = {page: (similarity_scores.get(page, 0) + auth_scores[page]) for page in pages}

    # Filter the final_scores to only include the keys in similarity_scores
    filtered_final_scores = {page: score for page, score in final_scores.items() if page in similarity_scores}

    # Sort the pages by their final_scores
    ranked_pages = dict(sorted(filtered_final_scores.items(), key=lambda x: x[1], reverse=True))
    ranked_pages = combine_similarity_pagerank(ranked_pages, child_links)

    return ranked_pages