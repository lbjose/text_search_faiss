from faiss import IndexFlatIP, write_index, read_index
from time import time

import tensorflow_hub as hub
import numpy as np
import json

from typing import List, Tuple

model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(model_url)


def create_faiss_ind(file_name: str = ""):
    """
        Function creates the Faiss index for the given data.

        We can use any models to generate embeddings, here Universal Sentence Encoder is used.

        Params:
            - data - list of questions to be searched on.
            - file_name - filename for the faiss index
    """
    embedded_data = model(data)
    index = IndexFlatIP(embedded_data[0].shape[0])
    index.add(np.array(embedded_data, dtype=np.float32))
    file_name = str(time) if not file_name else file_name
    write_index(index, file_name)


def search_query(index_path: str, query: List[str], k: int = 3) -> List[List[Tuple]]:
    """
        Given a query function returns the top k results.

        Params:
            - index_path - saved faiss index path
            - data - saved data
            - query - query to search, query can be a list of questions to be searched, as multiple queries can be
                      searched at the same time.
            - k - top k results to return

        Output:
            - matches_sents - list of list (sentence, score) pairs, at each index matching sentences for each query.
    """
    matched_sents = []
    query_embedding = model(query)
    index = read_index(index_path)
    top_k = index.search(np.array(query_embedding, dtype=np.float32), k)
    sents = data[top_k[1]]
    score = top_k[0]
    for match_pair, score_pair in zip(sents, score):
        matched_sents.append([(sent, score) for sent, score in zip(match_pair, score_pair)])
    return matched_sents


def search_query_in_range(index_path: str, query: List[str], threshold: int = 0.75) -> List[List[Tuple]]:
    """
        Given a query function returns the top k results.

        Params:
            - index_path - saved faiss index path
            - data - saved data
            - query - query to search, query can be a list of questions to be searched, as multiple queries can be
                      searched at the same time.
            - threshold - lower threshold score range

        Output:
            - matches_sents - list of list (sentence, score) pairs, at each index matching sentences for each query.
    """
    matched_sents = []
    start_id = 0
    query_embedding = model(query)
    index = read_index(index_path)
    inds, scores, data_pos = index.range_search(np.array(query_embedding, dtype=np.float32), thresh=threshold)
    scores = np.array(scores)
    print(inds, "\n", scores, "\n", data_pos)
    for ind, curr_id in enumerate(inds[1:]):
        if not start_id and not curr_id:
            start_id = curr_id
            matched_sents.append([])
            continue
        if not start_id and curr_id:
            matched_sents.append(
                list(sorted(zip(data[data_pos[0:curr_id]], scores[start_id:curr_id]), key=lambda x: x[1], reverse=True))
            )
            start_id = curr_id
            continue

        matched_sents.append(list(
            sorted(zip(data[data_pos[start_id:curr_id]], scores[start_id:curr_id]), key=lambda x: x[1], reverse=True))
        )
        start_id = curr_id
    return matched_sents


if __name__ == "__main__":
    global data
    data = json.load(open("data/sample.json", "r"))
    data = np.array(data)

    # Creating faiss index
    create_faiss_ind("covid")

    # # Example passing a single query for search
    query = ["covid"]
    matched_sents = search_query("covid", query, 4)
    print("Query:: ", query[0])
    print("Matched sents::", matched_sents[0])
    print("--"*30)

    # Example passing a multiple query for search
    queries = ["what is covid", "should I use face mask"]
    matched_sents = search_query("covid", queries, 4)
    for ind, each in enumerate(queries):
        print("Query:: ", each)
        print("Matched sents::", matched_sents[ind])
        print("--" * 30)
    print("--" * 30)

    # Example passing a multiple query for ranged search
    queries = ["symptoms of covid", "should I use face mask"]
    matched_sents = search_query_in_range("covid", queries, 0.5)
    for ind, each in enumerate(queries):
        print("Query:: ", each)
        print("Matched sents::", matched_sents[ind])
        print("--" * 30)
    print("--" * 30)