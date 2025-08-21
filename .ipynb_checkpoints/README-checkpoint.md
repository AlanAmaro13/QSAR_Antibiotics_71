
# QSAR Antibiotics

Hello there! In this repository I present the main results obtanined in a AI implementation in the prediction of Biological Activity from molecular descriptors of Antibiotics molecules. The workflow goes from a MultiLinear model to Non Linear (Deep Neural Networks). Feature selection was implemented using Mutual Information. 



## Main Results: 
- Feature Selection: piPC4 is the molecular descriptor with the highest value of Mutual Information.
- Linear Model: The best results were obtained with 5 molecular descriptors: SlogP-VSA6, C1SP2, CIC1, AATSC2se, GATS4are. Using 6-Fold Cross validaton, the results are an MAPE: 7.19 +- 1.23. And a Determination Coefficient: 0.62 +- 0.19. 
- Non Linear Model: The best resuts were obtained with 2 molecualr descriptors: piPC4 and GATS3i. Where the MAPE is: 7.11 +- 1.49 and the R2: 0.63 +- 0.19.

## Validated results from Non Linear Models:
* Goodness of fit: 0.506

* The results R2 for each fold are: 0.39 | 0.56| 0.71| 0.71| 0.64| 0.68| 0.01. Where the mean value is: 0.53 with a std of 0.24.

* The value Q2: 0.592

* The value Q2ext: 0.571


## Authors

- [Alan Amaro](https://www.linkedin.com/in/alanamaro/)
- [Dr. Erick Padilla](https://scholar.google.com/citations?user=qCaGKSsAAAAJ&hl=es)


## Appendix

All the data as well as the model results are available in: 

[Data & Models](https://drive.google.com/drive/folders/145f-mhFdcVsQCNL8XA7dSE6bpXo6bPHo?usp=sharing)

