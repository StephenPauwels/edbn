# Extended Dynamic Bayesian Network: Detecting Anomalies in Multidimensional Hybrid Business Process Logs</h1>

This repository contains all implementation files and experiments conducted for the Extended Dynamic Bayesian Networks introduced in
[1] and further extended in [3] where we try to detect anomalies in Business Process logs. Also work done in [2] can be found
in this repository.

All experiments in the papers can be reproduced using the files in the Experiments directory in the project.

## Project Structure
- [Bohmer](https://github.com/StephenPauwels/edbn/tree/master/Bohmer): Contains our own implementation of the Likelihood Graphs introduced by Bohmer et al in [4]
- [Nolle](https://github.com/StephenPauwels/edbn/tree/master/Nolle): Contains the original implementations used by Nolle et al in [5]
- [EDBN](https://github.com/StephenPauwels/edbn/tree/master/eDBN): Contains our implementation of our Extended Dynamic Bayesian Network model
- [Experiments](https://github.com/StephenPauwels/edbn/tree/master/Experiments): Contains all files used for the experiments found in [1] and [3]
- [Concept Drift](https://github.com/StephenPauwels/edbn/tree/master/Concept%20Drift): Contains the files used for the BPI Challenge 2018 [2]
- [Utils](https://github.com/StephenPauwels/edbn/tree/master/Utils): Some extra implementations regarding datastructures, preprocessing and data generation
- [Data](https://github.com/StephenPauwels/edbn/tree/master/Data): Data used for the experiments


## References
1. [Pauwels, Stephen, and Toon Calders. "An Anomaly Detection Technique for Business Processes based on Extended Dynamic Bayesian Networks." (2019)](http://adrem.uantwerpen.be/bibrem/pubs/PauwelsSAC19.pdf)
2. [Pauwels, Stephen, and Toon Calders. "Detecting and Explaining Drifts in Yearly Grant Applications." BPI Challenge 2018 (2018)](http://adrem.uantwerpen.be//bibrem/pubs/pauwels2018BPIC.pdf)
3. Pauwels, Stephen, and Toon Calders. "Detecting Anomalies in Hybrid Business Process Logs." (2019) (under review)
4. [Böhmer, Kristof, and Stefanie Rinderle-Ma. "Multi-perspective anomaly detection in business process execution events." OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". Springer, Cham, 2016.](https://eprints.cs.univie.ac.at/4785/1/cr.pdf)
5. [Nolle, Timo, et al. "BINet: Multi-perspective Business Process Anomaly Classification." arXiv preprint arXiv:1902.03155 (2019).](https://arxiv.org/pdf/1902.03155.pdf)
