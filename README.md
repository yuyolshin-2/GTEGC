
# A Data-Driven End-to-End Framework for Traffic Estimation with Cross-Network Generalization

<img width="1733" height="1125" alt="GTE-GC framework_v2" src="https://github.com/user-attachments/assets/f4d6f1f6-6e46-4fea-bbc9-4c1963c2e62c" />

This is a PyTorch implementation of Generalizable Traffic Estimation with Graph Contrastive objective (GTE-GC) introduced in the paper entitled "A Data-Driven End-to-End Framework for Traffic Estimation with Cross-Network Generalization."
The paper is currently under review in Transportation Research Part C - Emerging Technologies.

## Introduction
A major challenge in concurrent traffic estimation efforts is the network-wide estimation of traffic conditions in newly developed or data-scarce transportation networks where no historical traffic observations are available. Although recent advances in data-driven machine learning have yielded promising results for analyzing dynamic traffic data in operational contexts, their application in early-stage urban planning for regions with no prior data remains limited.

Most studies aim to predict short-term traffic based on the available historical observations and tend to overfit to local spatial patterns, limiting generalizability to new or unobserved urban regions. This restricts their utility in planning contexts, where traffic statistics must often be estimated in environments with no prior traffic observations or sensor infrastructure.

## Task Description 
**Transductive Traffic Estimation** refers to estimating traffic conditions on road segments within the same transportation network used for training. In this setting, the network of a single city is partitioned into mutually exclusive training and test subsets, allowing the model to learn from a portion of the city while being evaluated on the remaining, unobserved segments. In this task, we can assume that the training and testing sets are samples from the same underlying distribution, as they are drawn from the same transportation network.

**Inductive Traffic Estimation** involves predicting traffic conditions in an entirely different transportation network from the one used during training. The model is trained on the full network of one city and evaluated on the complete network of another city, which may differ significantly in urban form, road topology, and land-use characteristics.

## Method
<img width="2000" height="586" alt="ITSE-GC architecture_edited" src="https://github.com/user-attachments/assets/b46dc1fb-dacb-454c-8330-e349ac319068" />
Overall architecture

We also utilize temporal self-attention, spatial feed-forward layer, and graph contrastive objective
 
## Performance Comparison 
#### Datasets
- TomTom Speed Dataset: Probe vehicle speed dataset retrieved from TomTom Move portal. Speed from 10 European cities are utilized.
- Land use dataset: CORINE Land Cover data
- Transportation network data from TomTom

![networks data](https://github.com/user-attachments/assets/a9e4e3d3-3877-40b7-9c8d-b844d38d1831)


#### Results on the transductive task
<img width="508" height="440" alt="image" src="https://github.com/user-attachments/assets/f825ca48-849d-4b19-ab69-238b823efa8d" />

#### Results on the inductive task
<img width="496" height="446" alt="image" src="https://github.com/user-attachments/assets/f874cb90-8423-40c8-9797-f8f95ce27598" />


