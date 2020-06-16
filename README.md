# NucleosomeDensity
Predict the nucleosome density in yeast with deep neural network.

This project is aimed at using a convolutional network (CNN) to predict the nucleosome density in yeast using the DNA sequence as input. The CNN is used to make the link between a 2001 bp long window of DNA and the nucleosome density at the center of the window. The DNA window is turned into a one-hot-encoded vector. With a trained model in hand it is possible to predict the effect of a mutation or the nucleosome density on an artificially designed DNA casset.

An already trained model is available for prediction. It was trained on all the genome with the chromosome 16 being excluded. 

### How to use this project to train a neural net ?

- **Training a model:**

```NucleosomeDensity/Programme$ >>> python training.py -d DNA_directory -f nuc_occupancy_file.csv -o output_name -m model_name -s -z -ds -t 1 2 3 -v 4```

- **Predicting with a model:**

NucleosomeDensity/Programme$ >>> python prediction.py -d DNA_directory -f nuc_occupancy_file.csv -w model_weights_file -m model_name -s

- **Predicting the single mutation effect:** 

NucleosomeDensity/Programme$ >>> python mutazome.py -l window_length

### What are the arguments parsed ?

- **-d DNA_directory:** a directory in *Programme/seq_chr* that contains the DNA sequence. One hdf5 file per chromosome named as *chr\d+\.hdf5*.

- **-f nuc_occupancy_file.csv:** a csv file in *Programme/Start_data* with the columns named as chr, pos (facultative) and value. The chr column stand for the chromosome number, pos for the position of the nucleotid in its chromosome and value for the nucleosome occupancy.

- **-o output_name:** the name of the .hdf5 file that will contain the model weights after training (register in Results_nucleosome). This name should match *weights_.+\.hdf5*

- **-w model_weights_file:** the name of the hdf5 file that contains the weights of the model with which we want to predict the nucleosome occupancy on chr16 of S.cerevisiae (register in Results_nucleosome).

- **-model_name:** the name of the model we want to train. Could be cnn, cnn_deep, cnn_dilated, cnn_lstm.

- **-s:** if parsed indicates that the model we want to train or predict with is a seq2seq model.

- **-z:** if parsed indicates that we want to include the zeros in the training set (zeros are mainly non mappable regions, by default we do not take them in the training set as the value zeros is artificial).

- **-l window_length:** the effect of the single mutations are tested with the models in *Results_nucleosomes/Final_Results*, we just need to specify the window length used by the model (151, 301, 501, 1001, 1501 or 2001).

### What are the data needed to begin ?

- the DNA sequence should be stored in hdf5 format with one file per chromosome in a directory set in *Programme/seq_chr*
- the nucleosome occupancy should be stored in a csv file (see previous section) in *Programme/Start_data*

### How could I find my results ?

- the weights of the trained model will be register in *Results_nucleosomes* with the name parsed as arguments (that should match *weights_.+\.hdf5*).
- the prediction on chromosome 16 are register in *Results_nucleosomes* with the name *y_pred_.+\.npy* (with *.+* behind the same as the one in the name of the file that contains the weights of the model).
