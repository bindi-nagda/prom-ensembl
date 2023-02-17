# prom-ensembl: Hard Pattern Mining and Ensemble Learning for Detecting DNA Promoter Sequences
#### Bindi M. Nagda, Van Minh Nguyen, [Ryan T. White](https://www.ryantwhite.com/nets)

![alt text](https://github.com/bindi-nagda/prom-ensembl/blob/main/Promotor.jpg)

## Motivation: 
Accurate identification of DNA promoter sequences is of crucial importance in unraveling the underlying mechanisms
that regulate gene transcription. Initiation of transcription is controlled through regulatory transcription factors binding
to promoter core regions in the DNA sequence. Detection of promoter regions is necessary if we are to build genetic
regulatory networks for biomedical and clinical applications. We propose a novel ensemble learning technique using deep
recurrent neural networks with convolutional feature extraction and hard negative pattern mining to detect several types
of promoter sequences, including promoter sequences with the TATA-box and without the TATA-box, within DNA
sequences of both humans and mice. Using previously published results and extensive independent tests demonstrates
our method sets a new state of the art in all four categories for accurately and precisely recognizing the stretch of base
pairs that code for the promoter region within the DNA sequences.

## Data
[EPDNew Database](https://epd.epfl.ch/EPDnew_database.php)

## Results
Our method shows superiority to 4 other state-of-the-art models since it minimizes the rate of both false positives and false negatives. 
The model presented is unrivaled in multiple measures of performance including Matthews Correlation Coefficient (MCC), 
precision, sensitivity and specificity. Our model yields the best MCC values across all organisms, achieving a greater 
than 99% score for all organisms except humans with TATA where it achieves a 98.7% score. It goes on to achieve
$\geq$99% in 14 out of the 16 performance metrics evaluated.

## Contact 
[Go to contact information](https://www.linkedin.com/in/bindinagda/)