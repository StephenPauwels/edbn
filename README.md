# Extended Dynamic Bayesian Network: Probabilities in Business Processes</h1>

This repository contains all implementation files and experiments conducted for the Extended Dynamic Bayesian Networks introduced in
[1].
  
All code for detecting anomalies [3] and make Predictions [4, 5] in Business Process Logs can be found in this repository.
 
All experiments in the papers can be reproduced using the files in the Anomalies and Predictions directory in the project.

## Project Structure
- [Anomalies](https://github.com/StephenPauwels/edbn/tree/master/Anomalies): Contains all files used for the experiments found in [1] and [3]
- [Concept Drift](https://github.com/StephenPauwels/edbn/tree/master/Concept%20Drift): Contains the files used for the BPI Challenge 2018 [2]
- [Predictions](https://github.com/StephenPauwels/edbn/tree/master/Predictions): Contains all files used for the Prediction experiments for BPM 2020 [4] and ICPM 2020 [5]
- [EDBN](https://github.com/StephenPauwels/edbn/tree/master/eDBN): Contains our implementation of our Extended Dynamic Bayesian Network model
- [Utils](https://github.com/StephenPauwels/edbn/tree/master/Utils): Some extra implementations regarding datastructures, preprocessing and data generation
- [Data](https://github.com/StephenPauwels/edbn/tree/master/Data): Data used for the experiments
- [RelatedMethods](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods): Implementations of the other methods used to run the comparison experiments with
    - [Bohmer](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Bohmer): Contains our own implementation of the Likelihood Graphs introduced by Bohmer et al in [6]
    - [Camargo](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Camargo): Contains the slightly adapted implementation used in [8]
    - [DiMauro](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/DiMauro): Implementation used in [9]
    - [Lin](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Lin): Our implementation of the method described in [10]
    - [Nolle](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Nolle): Contains the original implementations used by Nolle et al in [7]
    - [Tax](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Tax): Adapted implementation used in [11]
    - [Pasquadibisceglie](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Pasquadibisceglie): Adapted implementation used in [12]
    - [Taymouri](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Taymouri): Adapted implementation used in [13]
    - [Premiere](https://github.com/StephenPauwels/edbn/tree/master/RelatedMethods/Premiere): Adapted implementation used in [14]

## References
1. [Pauwels, Stephen, and Toon Calders. "An Anomaly Detection Technique for Business Processes based on Extended Dynamic Bayesian Networks." (2019)](http://adrem.uantwerpen.be/bibrem/pubs/PauwelsSAC19.pdf)
2. [Pauwels, Stephen, and Toon Calders. "Detecting and Explaining Drifts in Yearly Grant Applications." BPI Challenge 2018 (2018)](http://adrem.uantwerpen.be//bibrem/pubs/pauwels2018BPIC.pdf)
3. [Pauwels, Stephen, and Toon Calders. "Detecting Anomalies in Hybrid Business Process Logs." Applied Computing Review, Volume 19  Issue 2  Page 18-30 (2019)](http://adrem.uantwerpen.be//bibrem/pubs/AcmAnomaly.pdf)
4. [Pauwels, Stephen, and Toon Calders. "Bayesian Network based Predictions of Business Processes." Accepted for BPM Forum 2020](https://www.researchgate.net/publication/342918314_Bayesian_Network_based_Predictions_of_Business_Processes)
5. Pauwels, Stephen, and Toon Calders. "Evaluating Next Event Predictions: What Could go Wrong?" Under review for ICPM2020.
6. [Böhmer, Kristof, and Stefanie Rinderle-Ma. "Multi-perspective anomaly detection in business process execution events." OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". Springer, Cham, 2016.](https://eprints.cs.univie.ac.at/4785/1/cr.pdf)
7. [Nolle, Timo, et al. "BINet: Multi-perspective Business Process Anomaly Classification." arXiv preprint arXiv:1902.03155 (2019).](https://arxiv.org/pdf/1902.03155.pdf)
8. Camargo, M., Dumas, M., Gonz ́alez-Rojas, O.: Learning accurate lstm models ofbusiness processes. In: International Conference on Business Process Management.pp. 286–302. Springer (2019)
9. Di  Mauro,  N.,  Appice,  A.,  Basile,  T.M.:  Activity  prediction  of  business  processinstances  with  inception  cnn  models.  In:  International  Conference  of  the  ItalianAssociation for Artificial Intelligence. pp. 348–361. Springer (2019)
10.  Lin, L., Wen, L., Wang, J.: Mm-pred: a deep predictive model for multi-attributeevent  sequence.  In:  Proceedings  of  the  2019  SIAM  International  Conference  onData Mining. pp. 118–126. SIAM (2019)
11.  Tax, N., Verenich, I., La Rosa, M., Dumas, M.: Predictive business process mon-itoring with lstm neural networks. In: International Conference on Advanced In-formation Systems Engineering. pp. 477–492. Springer (2017)
12. Pasquadibisceglie, V., Appice, A., Castellano, G., & Malerba, D. (2019, June). Using convolutional neural networks for predictive process analytics. In 2019 International Conference on Process Mining (ICPM) (pp. 129-136). IEEE.
13. Taymouri, F., La Rosa, M., Erfani, S., Bozorgi, Z. D., & Verenich, I. (2020). Predictive Business Process Monitoring via Generative Adversarial Nets: The Case of Next Event Prediction. arXiv preprint arXiv:2003.11268.
14. Pasquadibisceglie V., Appice A., Castellano G., Malerba D. (2020) Predictive Process Mining Meets Computer Vision. In: Fahland D., Ghidini C., Becker J., Dumas M. (eds) Business Process Management Forum. BPM 2020. Lecture Notes in Business Information Processing, vol 392. Springer, Cham.
