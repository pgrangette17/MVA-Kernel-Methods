# MVA-Kernel-Methods
 Here is my final project with Antonin Joly for the course Kernel Methods for ML, supervised by Pr. Julien Mairal. and Pr. Michael Arbel and Pr. Alessandro Rudi, as part of the MSc MVA 2022-2023.

## Project objective and conditions:

The task is to classify graphs representing molecules into two classes, indicating whether the molecules have a specific function or not. The goal is to experiment with graph kernels to obtain graph embeddings and then apply a classifier to them.

During this Kaggle project, we were not allowed to use the *scikit-learn* library and had to create our own classifiers using only convex optimization libraries such as *cvxopt*.

## Explored methods:
- **Kernels:**
    - Subtree-based kernels: Weisfeiler-Lehman subtree kernel, Weisfeiler-Lehman edge kernel
    - Path-based kernel: Random walk kernel, Shortest path kernel
    - Graphlet kernel
- **Classifiers:**
    - Kernel SVM
    - Kernel Logistic Regression

## Results:

The best result we achieved was with an AUC score of 0.80 using the Weisfeiler-Lehman edge kernel combined with the RBF kernel and the Logistic Regression classifier.

Check out our final report to view the benchmark of our results :
[View report](https://github.com/pgrangette17/MVA-Kernel-Methods/blob/main/Report_Kaggle_GRANGETTE_JOLY.pdf)
