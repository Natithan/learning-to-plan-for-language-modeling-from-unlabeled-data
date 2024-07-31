import argparse
import pickle
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

def load_predictions(file_path):
    with open(file_path, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def analyze_prediction_distribution(predictions):
    all_predicted_codes = [int(np.argmax(entry[3], axis=-1)) for entry in predictions]
    all_true_codes = [entry[4].item() for entry in predictions]
    
    predicted_code_counter = Counter(all_predicted_codes)
    true_code_counter = Counter(all_true_codes)

    for code_id in range(1024):
        predicted_code_counter.setdefault(code_id, 0)
        true_code_counter.setdefault(code_id, 0)
    
    # Calculate relative frequencies for predicted codes
    total_predicted = sum(predicted_code_counter.values())
    predicted_relative_frequencies = {code: count / total_predicted for code, count in predicted_code_counter.items()}
    
    # Calculate relative frequencies for true codes
    total_true = sum(true_code_counter.values())
    true_relative_frequencies = {code: count / total_true for code, count in true_code_counter.items()}
    
    # Combine and sort codes by relative frequencies for consistent plotting
    combined_codes = set(predicted_code_counter.keys()) | set(true_code_counter.keys())
    sorted_combined_codes = sorted(combined_codes)
    
    predicted_frequencies = [predicted_relative_frequencies.get(code, 0) for code in sorted_combined_codes]
    true_frequencies = [true_relative_frequencies.get(code, 0) for code in sorted_combined_codes]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    width = 0.35  # the width of the bars
    r1 = np.arange(len(sorted_combined_codes))
    r2 = [x + width for x in r1]
    
    plt.bar(r1, predicted_frequencies, color='skyblue', width=width, label='Predicted')
    plt.bar(r2, true_frequencies, color='orange', width=width, label='True')
    
    plt.xlabel('Codes')
    plt.ylabel('Relative Frequency')
    plt.title('Distribution of Predicted and True Codes')
    #plt.xticks([r + width/2 for r in range(len(sorted_combined_codes))], sorted_combined_codes)
    plt.legend()
    plt.savefig('predicted_true_codes_distribution.png')

    discrepancies = {code: abs(predicted_relative_frequencies.get(code, 0) - true_relative_frequencies.get(code, 0)) for code in combined_codes}
    sorted_discrepancies = sorted(discrepancies.items(), key=lambda item: item[1], reverse=True)
    
    print("\nCodes with the Largest Discrepancy in Relative Frequency:")
    for code, discrepancy in sorted_discrepancies[:5]:  # Adjust the slice for more or fewer codes
        print(f"Code: {code}, Discrepancy: {discrepancy:.4f}")

def analyze_random_entries(predictions, K, num_samples=10):
    random_entries = random.sample(predictions, num_samples)
    print("\nRandomly Selected Entries Analysis:")
    for entry in random_entries:
        _, _, context_text, policy_logits, true_code = entry
        probabilities = np.exp(policy_logits) / np.sum(np.exp(policy_logits), axis=-1, keepdims=True)
        probabilities = probabilities.squeeze()
        top_k_indices = np.argsort(probabilities)[-K:]
        print("\nContext Text:")
        print(context_text)
        print("Top K Greedy Codes:")
        for index in reversed(top_k_indices):
            print(f"Code {index} with probability {probabilities[index]:.4f}")
        true_code = true_code.item()
        print(f"True Code: {true_code}")
        true_code_rank = np.where(np.argsort(probabilities)[::-1] == true_code)[0][0] + 1
        print(f"Rank of True Code: {true_code_rank}")

def analyze_code_diversity(predictions):
    from collections import defaultdict

    # Group predictions by article for both predicted and oracle codes
    predictions_by_article = defaultdict(lambda: {'predicted': [], 'oracle': []})
    for entry in predictions:
        art_count, _, _, policy_logits, true_code = entry
        predicted_code = int(np.argmax(policy_logits, axis=-1))
        predictions_by_article[art_count]['predicted'].append(predicted_code)
        predictions_by_article[art_count]['oracle'].append(true_code.item())

    # Calculate and print diversity ratio for each article
    print("Code Diversity Ratios per Article:")
    diversity_ratios = {'predicted': [], 'oracle': []}
    for art_count, codes in predictions_by_article.items():
        for code_type in ['predicted', 'oracle']:
            unique_codes = len(set(codes[code_type]))
            total_codes = len(codes[code_type])
            diversity_ratio = unique_codes / total_codes if total_codes > 0 else 0
            diversity_ratios[code_type].append(diversity_ratio)
            print(f"Article {art_count} - {code_type.capitalize()} Codes: {diversity_ratio:.4f}")

    # Calculate and print the average diversity ratio
    for code_type in ['predicted', 'oracle']:
        average_diversity_ratio = sum(diversity_ratios[code_type]) / len(diversity_ratios[code_type])
        print(f"\nAverage {code_type.capitalize()} Code Diversity Ratio: {average_diversity_ratio:.4f}")

import numpy as np
from scipy.stats import entropy

def calculate_kl_divergence(predictions):
    # Extract oracle and predicted codes
    oracle_codes = [entry[4] for entry in predictions]
    predicted_codes = [np.argmax(entry[3], axis=-1) for entry in predictions]

    # Calculate the distribution of oracle and predicted codes
    oracle_distribution = np.bincount(oracle_codes, minlength=np.max(oracle_codes)+1)
    predicted_distribution = np.bincount(predicted_codes, minlength=np.max(predicted_codes)+1)

    # Normalize to get probabilities
    oracle_probs = oracle_distribution / np.sum(oracle_distribution)
    predicted_probs = predicted_distribution / np.sum(predicted_distribution)

    # Add a small value to avoid division by zero or log(0) in KL calculation
    epsilon = 1e-10
    oracle_probs = np.where(oracle_probs == 0, epsilon, oracle_probs)
    predicted_probs = np.where(predicted_probs == 0, epsilon, predicted_probs)

    # Calculate KL Divergence
    kl_divergence = entropy(oracle_probs, predicted_probs)

    print(f"KL Divergence (Oracle || Predicted): {kl_divergence:.4f}")

# Add a call to calculate_kl_divergence in the main function with an appropriate flag if needed

def analyze_longest_articles(predictions):
    from collections import defaultdict

    # Group predictions by article
    predictions_by_article = defaultdict(list)
    for entry in predictions:
        art_count = entry[0]
        predictions_by_article[art_count].append(entry)

    # Identify top 3 longest articles
    top_3_longest_articles = sorted(predictions_by_article.items(), key=lambda x: len(x[1]), reverse=True)[:3]

    # Print analysis for each of the top 3 longest articles
    for art_count, entries in top_3_longest_articles:
        print(f"\nAnalysis for Article {art_count}:")
        last_entry = entries[-1]
        context_text = last_entry[2]
        print("\nContext of the Last Entry:")
        print(context_text)

        predicted_codes = [int(np.argmax(entry[3], axis=-1)) for entry in entries]
        oracle_codes = [entry[4].item() for entry in entries]

        print("\nPredicted Codes vs Oracle Codes:")
        for predicted_code, oracle_code in zip(predicted_codes, oracle_codes):
            print(f"{predicted_code:<20} {oracle_code}")

def analyze_performance_metrics(predictions):
    ranks = []

    correct_predictions = 0

    for _, _, _, policy_logits, true_code in predictions:
        true_code = true_code.item()

        # Convert logits to probabilities
        probabilities = policy_logits # no need to normalize as we are only interested in the rank
        probabilities = probabilities.flatten()
        
        # Sort probabilities in descending order and get the indices
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Find the rank of the true code
        true_code_rank = np.where(sorted_indices == true_code)[0][0] + 1
        ranks.append(true_code_rank)
        
        # Update correct predictions count
        correct_predictions += (sorted_indices[0] == true_code)
    
    # Calculate average rank, median rank, and accuracy
    average_rank = np.mean(ranks) if ranks else 0
    median_rank = np.median(ranks) if ranks else 0
    accuracy = (correct_predictions / len(predictions)) * 100 if len(predictions) > 0 else 0

    print(f"Average Rank of Oracle Code: {average_rank:.2f}")
    print(f"Median Rank of Oracle Code: {median_rank:.2f}")
    print(f"Accuracy of Predictions: {accuracy:.2f}%")

import matplotlib.pyplot as plt

def plot_rank_histogram(predictions):
    ranks = []

    for _, _, _, policy_logits, true_code in predictions:
        true_code = true_code.item()

        # Convert logits to probabilities
        probabilities = policy_logits.flatten()
        
        # Sort probabilities in descending order and get the indices
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Find the rank of the true code
        true_code_rank = np.where(sorted_indices == true_code)[0][0] + 1
        ranks.append(true_code_rank)
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=range(1, 1025), density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Ranks with Relative Frequencies')
    plt.xlabel('Rank')
    plt.ylabel('Relative Frequency')
    plt.xlim(1, 1024)
    plt.xticks(np.arange(1, 1025, step=1023/10))  # Adjust the ticks to show a reasonable range
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('rank_histogram.png')


def analyze_consecutive_predictions(predictions, N=10):
    from collections import defaultdict

    # Group predictions by article
    predictions_by_article = defaultdict(list)
    for entry in predictions:
        art_count = entry[0]
        predictions_by_article[art_count].append(entry)

    # Initialize a counter for the cases printed
    cases_printed = 0

    # Analyze each article
    for art_count, entries in predictions_by_article.items():
        consecutive_count = 1  # Start with 1 to count the current code being analyzed
        last_code = None

        for i in range(len(entries)):
            current_code = int(np.argmax(entries[i][3], axis=-1))
            true_code = entries[i][4].item()

            # Check if the current code is the same as the last one
            if current_code == last_code:
                consecutive_count += 1
            else:
                consecutive_count = 1  # Reset if the current code is different

            # If we have found N consecutive predictions, print the analysis
            if consecutive_count == N:
                if cases_printed < 5:  # Limit to max 5 cases
                    print(f"\nArticle {art_count}, Sequence ending at prediction {i+1}:")
                    # Print the predicted codes next to the true codes for the sequence
                    for j in range(i-N+1, i+1):
                        pred_code = int(np.argmax(entries[j][3], axis=-1))
                        true_code = entries[j][4].item()
                        print(f"Predicted: {pred_code}, True: {true_code}")
                    # Print the context of the last predicted code
                    print("Context of the last predicted code:")
                    print(entries[i][2])
                    cases_printed += 1
                else:
                    return  # Stop the function if we have already printed 5 cases

            last_code = current_code

def main():
    parser = argparse.ArgumentParser(description="Analyze Predictions")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the pickle file containing predictions.")
    parser.add_argument("--K", type=int, default=5, help="Top K greedy codes to display.")
    parser.add_argument("--N", type=int, default=10, help="At least N consecutive predictions needed when analyzing consecutive patterns..")
    parser.add_argument("--random_entries", action="store_true", help="Perform random entries analysis.")
    parser.add_argument("--prediction_distribution", action="store_true", help="Perform prediction distribution analysis.")
    parser.add_argument("--code_diversity", action="store_true", help="Analyze code diversity within articles.")
    parser.add_argument("--longest_articles", action="store_true", help="Analyze the longest articles.")
    parser.add_argument("--performance_metrics", action="store_true", help="Compute performance metrics.")
    parser.add_argument("--kl_divergence", action="store_true", help="Compute KL Divergence between oracle and predicted codes.")
    parser.add_argument("--confidence_correlations", action="store_true", help="Compute correlation between avg rank and confidence metrics.")
    parser.add_argument("--consecutive_patterns", action="store_true", help="Consecutive patterns.")
    parser.add_argument("--rank_histogram", action="store_true", help="Rank histogram.")
    args = parser.parse_args()

    predictions = load_predictions(args.file_path)

    if args.prediction_distribution:
        analyze_prediction_distribution(predictions)
    if args.random_entries:
        analyze_random_entries(predictions, args.K)
    if args.code_diversity:
        analyze_code_diversity(predictions)
    if args.longest_articles:
        analyze_longest_articles(predictions)
    if args.performance_metrics:
        analyze_performance_metrics(predictions)
    if args.kl_divergence:
        calculate_kl_divergence(predictions)
    if args.confidence_correlations:
        compute_confidence_correlations(predictions)
    if args.consecutive_patterns:
        analyze_consecutive_predictions(predictions, args.N)
    if args.rank_histogram:
        plot_rank_histogram(predictions)

from scipy.stats import entropy, pearsonr

def calculate_confidence_metrics(predictions):
    ranks = []
    max_probs = []
    margins = []
    entropies = []

    for _, _, _, policy_logits, true_code in predictions:
        true_code = true_code.item()
        probabilities = np.exp(policy_logits) / np.sum(np.exp(policy_logits), axis=-1, keepdims=True)
        sorted_probs = np.sort(probabilities)
        sorted_indices = np.argsort(probabilities)
        sorted_probs = sorted_probs.flatten()
        sorted_indices = sorted_indices.flatten()
        sorted_probs = np.flip(sorted_probs)
        sorted_indices = np.flip(sorted_indices)

        # Rank of the true code
        true_code_rank = np.where(sorted_indices == true_code)[0] + 1
        ranks.append(true_code_rank[0])

        # Maximum probability
        max_prob = float(sorted_probs[0])
        max_probs.append(max_prob)

        # Margin
        margin = float(sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0])
        margins.append(float(margin))

        # Entropy
        prediction_entropy = entropy(probabilities[0])
        entropies.append(prediction_entropy)

    return ranks, max_probs, margins, entropies

def compute_correlations(ranks, max_probs, margins, entropies):
    # Compute correlation between average rank and each confidence metric
    avg_rank = np.mean(ranks)
    print(f"Correlation between Average Rank and Max Probability: {pearsonr(ranks, max_probs)[0]:.4f}")
    print(f"Correlation between Average Rank and Margin: {pearsonr(ranks, margins)[0]:.4f}")
    print(f"Correlation between Average Rank and Entropy: {pearsonr(ranks, entropies)[0]:.4f}")

def compute_confidence_correlations(predictions):
    ranks, max_probs, margins, entropies = calculate_confidence_metrics(predictions)
    compute_correlations(ranks, max_probs, margins, entropies)

if __name__ == "__main__":
    main()