import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm


def main(
    features_dir: str,
    output_dir: str,
    features_pattern: str = "*.pt",
    keep_percentage: float = 0.1,
    k: int = 100,
    n_iter: int = 300,
):
    # create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define output paths
    init_centroids_path = Path(output_dir) / f"initial-centroids-k-{k}.pt"
    objective_path = Path(output_dir) / f"objective-k-{k}.csv"
    centroids_path = Path(output_dir) / f"centroids-k-{k}.pt"

    features_paths = sorted(list(Path(features_dir).glob(features_pattern)))

    print(f"Found {len(features_paths)} .pt files to load")

    # count the number of features
    num_train_features = 0
    feature_dim = None
    for features_path in tqdm(features_paths, desc="Counting features"):
        f = torch.load(features_path).numpy()
        N, D = f.shape
        n = int(N * keep_percentage)
        num_train_features += n
        if feature_dim is None:
            feature_dim = D
        else:
            assert feature_dim == D, "Feature dimension mismatch"

    print(f"Total number of features: {num_train_features}")
    print(f"Creating features array of shape: ({num_train_features}, {feature_dim})")

    features = np.empty((num_train_features, feature_dim), dtype=np.float32)
    i = 0
    for features_path in tqdm(features_paths, desc="Loading features"):
        f = torch.load(features_path).numpy()
        N, _ = f.shape
        n = int(N * keep_percentage)
        indices = np.random.choice(N, size=n, replace=False)
        features[i : i + n] = f[indices]
        i += n

    # INITIALIZE CLUSTERS WITH K-MEANS++
    print("Initializing centers with k-means++...")
    init_centers, init_indices = kmeans_plusplus(
        X=features,
        n_clusters=k,
    )

    # SAVE THE INITIAL CENTROIDS
    torch.save(torch.from_numpy(init_centers), init_centroids_path)
    print(f"Saved initial centroids to {init_centroids_path}")

    # TRAIN K-MEANS
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=k,
        niter=n_iter,
        verbose=True,
        spherical=False,
        max_points_per_centroid=1_000_000,
    )
    print("Training k-means...")
    kmeans.train(features, init_centroids=init_centers)

    # kmeans.obj is a list of the objective function values at each iteration
    # save this to a csv file with two columns: iteration and objective
    np.savetxt(
        objective_path,
        np.array(kmeans.obj).reshape(-1, 1),
        delimiter=",",
    )
    print(f"Saved objective function to {objective_path}")

    # GET THE CENTROIDS
    centroids = kmeans.centroids

    # SAVE THE CENTROIDS
    torch.save(torch.from_numpy(centroids), centroids_path)
    print(f"Saved centroids to {centroids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a k-means model on .pt features."
    )
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.pt")
    parser.add_argument("--keep-percentage", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    main(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        features_pattern=args.pattern,
        keep_percentage=args.keep_percentage,
        k=args.k,
        n_iter=args.n_iter,
    )
