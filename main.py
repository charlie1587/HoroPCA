"""Run dimensionality reduction experiment."""

import argparse
import logging
import os

import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import torch
from scipy.stats import gaussian_kde

import geom.hyperboloid as hyperboloid
import geom.poincare as poincare
from learning.frechet import Frechet
from learning.pca import TangentPCA, EucPCA, PGA, HoroPCA, BSA
from utils.data import load_graph, load_embeddings
from utils.metrics import avg_distortion_measures, compute_metrics, format_metrics, aggregate_metrics
from utils.sarkar import sarkar, pick_root

parser = argparse.ArgumentParser(
    description="Hyperbolic dimensionality reduction"
)
parser.add_argument('--dataset', type=str, help='which datasets to use', default="smalltree",
                    choices=["smalltree", "phylo-tree", "bio-diseasome", "ca-CSphd", "bioscan_taxonomy"])
parser.add_argument('--model', type=str, help='which dimensionality reduction method to use', default="horopca",
                    choices=["pca", "tpca", "pga", "bsa", "hmds", "horopca"])
parser.add_argument('--metrics', nargs='+', help='which metrics to use', default=["distortion", "frechet_var"])
parser.add_argument(
    "--dim", default=10, type=int, help="input embedding dimension to use"
)
parser.add_argument(
    "--n-components", default=2, type=int, help="number of principal components"
)
parser.add_argument(
    "--save-plot", action="store_true", help="save 2D projection plot and coordinates"
)
parser.add_argument(
    "--outdir", type=str, default="results", help="directory to save plots/coords"
)

parser.add_argument(
    "--lr", default=5e-2, type=float, help="learning rate to use for optimization-based methods"
)
parser.add_argument(
    "--n-runs", default=5, type=int, help="number of runs for optimization-based methods"
)
parser.add_argument('--use-sarkar', default=False, action='store_true', help="use sarkar to embed the graphs")
parser.add_argument(
    "--sarkar-scale", default=3.5, type=float, help="scale to use for embeddings computed with Sarkar's construction"
)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    pca_models = {
        'pca': {'class': EucPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'tpca': {'class': TangentPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'pga': {'class': PGA, 'optim': True, 'iterative': True, "n_runs": args.n_runs},
        'bsa': {'class': BSA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
        'horopca': {'class': HoroPCA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
    }
    metrics = {}
    embeddings = {}
    logging.info(f"Running experiments for {args.dataset} dataset.")

    # load a graph args.dataset
    graph = load_graph(args.dataset)
    n_nodes = graph.number_of_nodes()
    nodelist = np.arange(n_nodes)
    graph_dist = torch.from_numpy(nx.floyd_warshall_numpy(graph, nodelist=nodelist))
    logging.info(f"Loaded {args.dataset} dataset with {n_nodes} nodes")

    # get hyperbolic embeddings
    if args.use_sarkar:
        # embed with Sarkar
        logging.info("Using sarkar embeddings")
        root = pick_root(graph)
        z = sarkar(graph, tau=args.sarkar_scale, root=root, dim=args.dim)
        z = torch.from_numpy(z)
        z_dist = poincare.pairwise_distance(z) / args.sarkar_scale
    else:
        # load pre-trained embeddings
        logging.info("Using optimization-based embeddings")
        # Remove dimension restriction for custom datasets
        if args.dataset not in ["smalltree", "phylo-tree", "bio-diseasome", "ca-CSphd"]:
            logging.info(f"Loading custom dataset {args.dataset} with {args.dim} dimensions")
        else:
            assert args.dim in [2, 10, 50], "pretrained embeddings are only for 2, 10 and 50 dimensions"
        z = load_embeddings(args.dataset, dim=args.dim)
        z = torch.from_numpy(z)
        z_dist = poincare.pairwise_distance(z)
    if torch.cuda.is_available():
        z = z.cuda()
        z_dist = z_dist.cuda()
        graph_dist = graph_dist.cuda()

    # compute embeddings' distortion
    distortion = avg_distortion_measures(graph_dist, z_dist)[0]
    logging.info("Embedding distortion in {} dimensions: {:.4f}".format(args.dim, distortion))

    # Compute the mean and center the data
    logging.info("Computing the Frechet mean to center the embeddings")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(z, return_converged=True)
    logging.info(f"Mean computation has converged: {has_converged}")
    x = poincare.reflect_at_zero(z, mu_ref)

    # Run dimensionality reduction methods
    logging.info(f"Running {args.model} for dimensionality reduction")
    metrics = []
    dist_orig = poincare.pairwise_distance(x)
    # holder for last-run 2D embedding (Poincaré ball)
    proj_2d = None
    if args.model in pca_models.keys():
        model_params = pca_models[args.model]
        for _ in range(model_params["n_runs"]):
            model = model_params['class'](dim=args.dim, n_components=args.n_components, lr=args.lr, max_steps=500)
            if torch.cuda.is_available():
                model.cuda()
            model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
            metrics.append(model.compute_metrics(x))
            # map_to_ball returns the low-dim Poincaré coords; keep last run for plotting
            embeddings = model.map_to_ball(x)
            proj_2d = embeddings.detach().cpu().numpy()
        metrics = aggregate_metrics(metrics)
    else:
        # run hMDS baseline
        logging.info(f"Running hMDS")
        x_hyperboloid = hyperboloid.from_poincare(x)
        distances = hyperboloid.distance(x.unsqueeze(-2), x.unsqueeze(-3))
        D_p = poincare.pairwise_distance(x)
        x_h = hyperboloid.mds(D_p, d=args.n_components)
        x_proj = hyperboloid.to_poincare(x_h)
        proj_2d = x_proj.detach().cpu().numpy()
        metrics = compute_metrics(x, x_proj)
    logging.info(f"Experiments for {args.dataset} dataset completed.")
    logging.info("Computing evaluation metrics")
    results = format_metrics(metrics, args.metrics)
    for line in results:
        logging.info(line)

    # --------------------------
    # Save coordinates & a plot
    # --------------------------
    if args.save_plot:
        if args.n_components != 2:
            logging.warning("save-plot requested but n-components != 2; skipping plot.")
        else:
            os.makedirs(args.outdir, exist_ok=True)
            base = f"{args.dataset}_{args.model}_dim{args.dim}_nc{args.n_components}"

            # Save coords as .npy and .csv
            npy_path = os.path.join(args.outdir, base + "_coords.npy")
            csv_path = os.path.join(args.outdir, base + "_coords.csv")
            np.save(npy_path, proj_2d)
            np.savetxt(csv_path, proj_2d, delimiter=",")
            logging.info(f"Saved coordinates to: {npy_path} and {csv_path}")

            # Enhanced scatter plot with density and better labels
            fig_path = os.path.join(args.outdir, base + ".png")
            plt.figure(figsize=(10, 10))
            
            # Create density plot in background
            from scipy.stats import gaussian_kde
            if len(proj_2d) > 1:
                # Create density estimation
                kde = gaussian_kde(proj_2d.T)
                
                # Create grid for density plot
                xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 100), 
                                   np.linspace(-1.1, 1.1, 100))
                grid_coords = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(grid_coords).reshape(xx.shape)
                
                # Plot density as background
                plt.contourf(xx, yy, density, levels=20, alpha=0.3, cmap='Blues')
            
            # Plot points
            plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=30, alpha=0.8, 
                       c='darkblue', edgecolors='white', linewidth=0.5)
            
            # Add labels to each point (using node indices for now)
            # For better visualization, only label points not too close to others
            labeled_points = set()
            min_distance = 0.1  # Minimum distance between labels
            
            for i in range(len(proj_2d)):
                # Check if this point is too close to already labeled points
                too_close = False
                for j in labeled_points:
                    dist = np.sqrt((proj_2d[i, 0] - proj_2d[j, 0])**2 + 
                                 (proj_2d[i, 1] - proj_2d[j, 1])**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    # Add label with node index (you can replace this with actual names)
                    label_text = f"node_{i}"  # Replace with actual node names if available
                    plt.annotate(label_text, (proj_2d[i, 0], proj_2d[i, 1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=9, alpha=0.9, 
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
                    labeled_points.add(i)
            
            # draw unit circle for reference (Poincaré ball)
            circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", 
                              linewidth=2, color='black', alpha=0.8)
            ax = plt.gca()
            ax.add_artist(circle)
            
            # Set axis limits to show complete circle with some padding
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal", adjustable="box")
            
            # Add labels and title
            plt.xlabel("Poincaré Coordinate 1", fontsize=12)
            plt.ylabel("Poincaré Coordinate 2", fontsize=12)
            plt.title(f"{args.model} on {args.dataset}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved plot to: {fig_path}")
