# DSEMSM
This repository is the official implementation of [Accelerating Time Series Similarity Search under Move-Split-Merge Distance via Dissimilarity Space Embedding], submitted to ESWA （under review））
# Accelerating Time Series Similarity Search under Move-Split-Merge Distance via Dissimilarity Space Embedding
This repository is the official implementation of [Accelerating Time Series Similarity Search under Move-Split-Merge Distance via Dissimilarity Space Embedding], submitted to ESWA （under review））

## Usage
download and read the UCR dataset(https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

The MSM lower bound GLB is used in DSEMSM. Please refer to [Accelerating Similarity Search for Elastic Measures: A Study and New Generalization of Lower Bounding Distances]

To evaluate the retrieval accuracy of the proposed DSEMSM method, run 'accuracy.py'

To evaluate the running time of the proposed DSEMSM method, run 'efficiency.py'

To demonstrate the necessary of the verification stage, run 'necessary of verifivation.py'. Change k value and record the results

The Parameter Sensitivity Analysis can be verified by changing the parameters used in 'accuracy.py' and 'efficiency.py'.


## General Example
Two datasets 'CBF' and 'Wafer' are provided. Just run 'accuracy.py' and 'efficiency.py'

**Requirements**: NumPy, scipy, matplotlib, math



