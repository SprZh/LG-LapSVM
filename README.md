# LG-LapSVM
A Lie Group Laplacian Support Vector Machine for Semi-Supervised Learning

## Abatract
As a semi-supervised learning (SSL) method, Laplacian Support Vector Machine (LapSVM) utilizes both labeled and unlabeled data to form a manifold regularization with graph theory and often receives a better performance than other SVM methods when the labeled data samples are insufficient. However, current LapSVMs often struggle to handle high-dimensional data with transformation invariance such as images and videos, ignore the neighboring and distribution information of labeled data samples, and are sensitive to data samples around the decision boundary. Aiming at these limitations, this paper proposes a novel SVM method for SSL referred to as LG-LapSVM by incorporating Lie group theory, the theory of local behavior similarity, and RoBoSS
loss function to the existing LapSVM framework. LG-LapSVM first maps data samples from Euclidean space to a Lie group manifold where a Laplacian graph is built according to geodesic distances between group elements and neighboring and distribution information of labeled data samples; then forms the Lie group manifold regulation by using the Laplacian matrix generated from the Laplacian graph and Lie group kernel metrics between each pair of data samples; finally trains a SVM classifier with an objective function including a RoBoSS loss function for labeled data samples and the Lie group manifold regulation. The method was validated against a variety of datasets and the classification performances were evaluated with different metrics through a systematic comparison with nine typical SVM methods for SSL. The experiment results demonstrate the effectiveness and the superiority of the proposed method especially for image datasets.
