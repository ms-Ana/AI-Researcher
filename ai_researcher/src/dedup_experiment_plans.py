from nltk.corpus import stopwords
import string
import json
from collections import Counter
import numpy as np
import argparse
import os
import shutil
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
    plt.bar(x, y, color="skyblue")
    plt.xlabel("Number of Occurrences")
    plt.ylabel("Number of Strings")
    plt.title("Frequency of String Occurrences")
    plt.xticks(x)
    plt.grid(axis="y")
    plt.show()


def process_text(input_text, tokenize=False):
    # Define the list of stopwords
    stop_words = set(stopwords.words("english"))

    # Lowercase the input text
    lowercased_text = input_text.lower()

    # Remove punctuation from the text
    no_punctuation_text = lowercased_text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize the text into words
    words = no_punctuation_text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a single string
    processed_text = " ".join(filtered_words)

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
    parser.add_argument("--cache_dir", type=str, default="bias", help="cache file name")
    parser.add_argument(
        "--cache_name", type=str, default="bias", help="cache file name"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="NN Similarity Threshold",
    )
    parser.add_argument(
        "--dedup_cache_dir", type=str, default="bias", help="cache file name"
    )
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Similarity Threshold: ", args.similarity_threshold)

    all_ideas = []
    all_filenames = []
    filenames = os.listdir(os.path.join(args.cache_dir, args.cache_name))
    filenames = [f for f in filenames if f.endswith(".json")]

    for filename in filenames:
        with open(os.path.join(args.cache_dir, args.cache_name, filename), "r") as f:
            paper = json.load(f)
        if "full_experiment_plan" in paper and isinstance(
            paper["full_experiment_plan"], dict
        ):
            try:
                all_filenames.append(filename)
            except:
                continue

    print("#ideas: ", len(all_filenames))

    similarity_matrix = np.load(
        os.path.join(args.cache_dir, args.cache_name + "_similarity_matrix.npy")
    )
    if len(similarity_matrix) != len(all_filenames):
        print("Error: similarity matrix size mismatch")
        exit(0)

    final_ideas = []
    filter_idx = []  ## ideas that should be filtered
    for i in range(len(all_filenames)):
        if i not in filter_idx:
            ## add current idea to filtered_ideas
            final_ideas.append(all_filenames[i])

            ## filter out similar ideas
            for j in range(i + 1, len(all_filenames)):
                if (
                    j not in filter_idx
                    and similarity_matrix[i][j] > args.similarity_threshold
                    or all_filenames[j] == all_filenames[i]
                ):
                    filter_idx.append(j)

    print("#final ideas: ", len(final_ideas))

    ## copy final ideas from cache_dir to dedup_cache_dir
    dedup_cache_dir = os.path.join(args.dedup_cache_dir, args.cache_name)
    if not os.path.exists(dedup_cache_dir):
        os.makedirs(dedup_cache_dir)
    for filename in final_ideas:
        shutil.copy(
            os.path.join(args.cache_dir, args.cache_name, filename),
            os.path.join(args.dedup_cache_dir, args.cache_name, filename),
        )
