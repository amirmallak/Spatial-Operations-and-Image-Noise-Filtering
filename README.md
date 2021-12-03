# Spatial-Operations-and-Image-Noise-Filtering
Implementing routines that performs spatial operations and building 2D Conv filters for denoising

This project includes:
* Adding Salt & Pepper Noise to images
* Adding Gaussian Noise (with controllable distribution) to images
* Filtering by 2D Convulotion Median Filter
* Filtering by 2D Convulotion Mean-Gaussian Filter (radius order derivative 2D mask)
* Bilateral Filtering - replacing pixels with euclidean metric weights of its neighbors, where the weights are determined according to the spatial and photometric (intensity) distances.

I compare methods with different Noise distributions, apply different filters, and conclude regarding matching a filter.
