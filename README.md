# machine-learning-on-sequence-data
This folder contains a variety of files for classification of virus genome sequences using machine learning. The genomes were processed using integer chaos game representation (iCGR) representation (as outlined in Numerical encoding of DNA sequences by chaos game representation with application in similarity comparison by Hoang, Yin and Yau (2016)) using the corresponding files in my Github Seq folder (please refer to the ReadMe for further directions).

- Initially, classification was tested with neural networks of varying complexity (Model 1 to 4 files). Classification even with 3 layers was fairly successful (>93% accuracy). 

- To get to the bottom of why classification was so easily achieved, Principal Component Analysis (PCA) was conducted on the even-scaled power signals (code included in the PCA files) and the variance was analysed by assessing variance over number of principal components (PCA selecting component number.py). 

- Overall and as can be seen in PCA 2D.png , PCA of power signals with 2 principal components was sufficient to separate sequences sufficiently for even multiclass logistic regression to classify sequences. The only problems arose with separation of H1N1 and H5N1 strains of influenza (Blue and Red on the PCA image) which is expected, since they have the same Neuraminidase.
