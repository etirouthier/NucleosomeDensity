# NucleosomeDensity
Predict the nucleosome density in yeast with deep neural network.

This project is aimed at predicting the nucleosome lanscape in yeast from the DNA sequence. 
We train a convolutional neural network to do so.
We apply a rolling window to the DNA sequence (window length : 2001 bp) and we try to predict the nucleosome occupancy 
at the center nucleotid (another way called seq2seq is to predict direclty the nucleosome landscape at 
every position of the window). The DNA sequence is turned into a vector wit a one-hot-encoding and passed as input of the CNN model.

The training is made on the 15 first chromosomes of S.cerevisiae and the prediction are made on its 16th. We want also to be able to
predict the effect of single mutation on the nucleosome positioning. To do so we can predict the nucleosome occupancy of an 
artificially mutated chromosome 16. In practice, every position on chromosome 16 is tested.

### How to use this project to train a neural net ?

- Training a model: NucleosomeDensity/Programme$ >>> python training.py -d DNA_directory -f nuc_occupancy_file.csv -o output_name -m model_name
- Predicting with a model: NucleosomeDensity/Programme$ >>> python prediction.py -d DNA_directory -f nuc_occupancy_file.csv -w model_weights_file -m model_name
- Predicting the single mutation effect: NucleosomeDensity/Programme$ >>> python mutazome.py -l window_length
