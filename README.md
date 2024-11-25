# Precision, Recall, F-score, and Jaccard Index for continuous, ratio-scale measurements
Variants of four agreement measures (Jaccard, Precision, Recall and F-score), applicable to data representing estimates of attributes at the ratio scale.
These measures can be used to estimate closeness of the continuous (cont.) attribute magnitude estimates and serve as viable equivalents of their common categorical counterparts.

## About
Gridded data representing attribute estimates at ratio scale are increasingly common for modelling spatial-environmental variables, including class area estimates (e.g. built-up surface area), population abundance (e.g. number of inhabitants), or vegetation-related measurements (e.g. canopy height). The accuracy of model-based gridded data, including classifications of remotely-sensed data, is usually assessed with measures based on confusion matrices with site-specific class allocations. Yet, these measures can only be applied to categorical attributes, not to ratio-scale attributes. Here, we introduce an approach to extend commonly used agreement measures estimated from a confusion matrix (Precision, Recall, F-score and Jaccard index) to non-negative ratio-scale attributes. 

The data provided in this repository allows for computing the measures of agreement between gridded datasets, stored as numpy arrays ar R matrices. Code examples showcase how to compute measures of error (Mean Error and Mean Absolute Error), measures of assositation (Pearson's correlation coefficient and slope of linear regression) and the four introdcued measures of sgreement: Precision, cont. Recall, cont. Jaccard and cont. F1-score, between two 5x5 grids, storing randomly generated non-negative continuous values.

![continuous agreement measures](https://github.com/katarzynagoch/PRERION/blob/main/measures.jpg)

## Installation
No installation required. The script Python-continuous-measures.py can be run in any Python environment (tested in Python 3.9 only). Dependencies (i.e., matplotlib, numpy, random) must be available. The script R-continuous-measures.R can be run in any Python environment (tested in R 4.4.2), no additional dependencies are required.

## Usage
Cont. Precision, cont. Recall, cont. Jaccard and cont. F1-score are suitable measures for estimating the accuracy of gridded datasets representing unevenly distributed and dispersed attributes at the ratio scale. They can be used for comparing gridded datasets of attributes at the ratio scale, which include absolute or relative estimates (e.g. canopy height or built-up surface density), but can also be applied to any non-spatial, ratio-scale data.  

## Publication
Pre-print available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4865121

## Contact
JRC-GHSL-DATA@ec.europa.eu

## Authors and acknowledgment
Katarzyna Krasnodębska, Martino Pesaresi, Unit E.1, Joint Research Centre, European Commission, 2024.

## License
XXX
