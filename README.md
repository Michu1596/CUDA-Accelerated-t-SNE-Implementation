# CUDA-Accelerated t-SNE Implementation

## Overview
t-SNE (t-distributed Stochastic Neighbor Embedding) is a machine learning algorithm for dimensionality reduction, particularly effective for visualizing high-dimensional data in 2D or 3D space. This implementation leverages CUDA C for parallel processing on NVIDIA GPUs, significantly improving performance for large datasets.

## Algorithm Description
t-SNE works by converting similarities between data points into joint probabilities and minimizing the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

### Key Features
- Non-linear dimensionality reduction
- Preservation of local structure
- Emphasis on revealing clusters and patterns
- GPU-accelerated computation


## License
MIT License - See LICENSE file for details.