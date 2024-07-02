import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
from collections import Counter
import numpy as np
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer

def plot_string_occurrences(strings_list):
    # Count occurrences of each string
    occurrences = Counter(strings_list)
    
    # Count how many strings have each occurrence count
    count_of_occurrences = Counter(occurrences.values())
    
    # Extracting the data for plotting
    x = sorted(count_of_occurrences.keys())
    y = [count_of_occurrences[occ] for occ in x]
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Strings')
    plt.title('Frequency of String Occurrences')
    plt.xticks(x)
    plt.grid(axis='y')
    plt.show()

def process_text(input_text, tokenize=False):
    # Define the list of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Lowercase the input text
    lowercased_text = input_text.lower()
    
    # Remove punctuation from the text
    no_punctuation_text = lowercased_text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text into words
    words = no_punctuation_text.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the filtered words back into a single string
    processed_text = ' '.join(filtered_words)

    if tokenize:
        return set(filtered_words)
    else:
        return process_text

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def find_representative_paper(cluster, similarity_matrix, labels):
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
    cluster_sims = similarity_matrix[cluster_indices][:, cluster_indices]
    avg_sims = cluster_sims.mean(axis=1)
    representative_index = cluster_indices[avg_sims.argmax()]
    return representative_index

def find_top_n_papers(representative_index, similarity_matrix, n=5):
    sims = similarity_matrix[representative_index]
    closest_indices = np.argsort(-sims)[:n]  # Sort in descending order and get top-n
    return closest_indices

def concatenate_idea(idea_k, idea_v):
    output = ""
    output += idea_k + "\n"
    output += "Problem: " + idea_v["Problem"] + "\n"
    output += "Existing Methods: " + idea_v["Existing Methods"] + "\n"
    output += "Motivation: " + idea_v["Motivation"] + "\n"
    output += "Proposed Method: " + idea_v["Proposed Method"] + "\n"
    output += "Experiment Plan: " + idea_v["Experiment Plan"] + "\n"

    return output


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_name', type=str, default="uncertainty_prompting", help='cache file name')
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_ideas = []
    all_idea_ks = []
    all_idea_vs = []
    topic = ""
    with open("../cache_results_claude_may/ideas_1k_claude3-5/{}.json".format(args.cache_name), "r") as f:
        ideas_json = json.load(f)
        topic = ideas_json["topic_description"]
        for ideas_dict in ideas_json["ideas"]:
            for idea_k, idea_v in ideas_dict.items():
                all_ideas.append(concatenate_idea(idea_k, idea_v))
                all_idea_ks.append(idea_k)
                all_idea_vs.append(idea_v)
    
    # all_ideas = all_ideas[:40]
    print ("#original ideas: ", len(all_ideas))

    embeddings = model.encode(all_ideas)
    similarity_matrix = model.similarity(embeddings, embeddings)
    similarity_matrix = similarity_matrix.numpy()
    np.fill_diagonal(similarity_matrix, 0)

    nn_similarity = []
    nn_similarity_idx = []
    avg_similarity = []
    for i in range(len(all_ideas)):
        nn_similarity.append(np.max(similarity_matrix[i]))
        nn_similarity_idx.append(np.argmax(similarity_matrix[i]))
        avg_similarity.append(np.sum(similarity_matrix[i]) / (len(all_ideas) - 1))

    avg_nn_similarity = np.mean(nn_similarity)
    print ("Avg NN Similarity: ", avg_nn_similarity)

    final_ideas = {}
    filter_idx = [] ## ideas that should be filtered
    for i in range(len(all_ideas)):
        if i not in filter_idx:
            ## add current idea to filtered_ideas
            final_ideas[all_idea_ks[i]] = all_idea_vs[i]

            ## filter out similar ideas
            for j in range(i+1, len(all_ideas)):
                if j not in filter_idx and similarity_matrix[i][j] >= avg_nn_similarity:
                    filter_idx.append(j)
    
    print ("#final ideas: ", len(final_ideas))

    final_json = {}
    final_json["topic_description"] = topic 
    final_json["ideas"] = final_ideas 
    with open("../cache_results_claude_may/ideas_1k_claude3-5_dedup/{}.json".format(args.cache_name), "w") as f:
        json.dump(final_json, f, indent=4)