import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def complex_to_real_vector(arr):
    return np.concatenate([arr.real.flatten(), arr.imag.flatten()])

def absolute_magnitude_vector(arr):
    return np.abs(arr).flatten()

def real_only_vector(arr):
    return arr.real.flatten()

# Files to compare
file1 = "reconsdata/brecons_og_2099_40.npy"
file2 = "reconsdata/brecons_og_2099_85.npy"

try:
    array1 = np.load(file1)
    array2 = np.load(file2)

    # Compute similarities
    complex_similarity = cosine_similarity([complex_to_real_vector(array1)],
                                           [complex_to_real_vector(array2)])[0][0]

    abs_magnitude_similarity = cosine_similarity([absolute_magnitude_vector(array1)],
                                                  [absolute_magnitude_vector(array2)])[0][0]

    real_only_similarity = cosine_similarity([real_only_vector(array1)],
                                             [real_only_vector(array2)])[0][0]

    # Print results
    print(f"Complex Data Similarity: {complex_similarity:.4f}")
    print(f"Absolute Magnitude Similarity: {abs_magnitude_similarity:.4f}")
    print(f"Real Values Only Similarity: {real_only_similarity:.4f}")

    # Plotting
    labels = ["Complex", "Absolute Magnitude", "Real Only"]
    similarities = [complex_similarity, abs_magnitude_similarity, real_only_similarity]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, similarities, color=['#4CAF50', '#2196F3', '#FFC107'])
    plt.title('Similarity Comparison')
    plt.ylabel('Cosine Similarity Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i, v in enumerate(similarities):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    plt.show()

except Exception as e:
    print(f"Error comparing files {file1} and {file2}: {e}")