# Region Tokenization

This module implements the logic for partitioning high-dimensional hand joint data into discrete spatial "regions" using vector quantization. This is a key preprocessing step for the Position LLM.

## Subdirectories

### [cluster/](cluster/)
Contains K-Means clustering implementations used to define the region centroids.

### [position_llm/](position_llm/)
Model definitions and datasets that specifically use region-based tokenization for hand movement prediction.

### [data_collection/](data_collection/)
Local utilities for processing data specifically for region tokenization.
