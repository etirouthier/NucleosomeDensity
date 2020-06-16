# NucleosomeDensity
Predict the nucleosome density in yeast with deep neural network.

This project is aimed at using a convolutional network (CNN) to predict the nucleosome density in yeast using the DNA sequence as input. The CNN is used to make the link between a 2001 bp long window of DNA and the nucleosome density at the center of the window. The DNA window is turned into a one-hot-encoded vector. With a trained model in hand it is possible to predict the effect of a mutation or the nucleosome density on an artificially designed DNA casset.

An already trained model is available for prediction. It was trained on all the genome with the chromosome 16 being excluded. 

### How to use this project to train a neural net ?

- **Training a model:**

```NucleosomeDensity/Programme$ >>> python training.py -d DNA_directory -f nuc_occupancy_file.csv -o weights_output.hdf5 -m model_name -s -z -ds -t 1 2 3 -v 4```

- **-d DNA_directory:** a directory in *Programme/seq_chr* that contains the DNA sequence. One hdf5 file per chromosome named as *chr\d+\.hdf5*.

- **-f nuc_occupancy_file.csv:** a csv file in *Programme/Start_data* with the columns named as chr, pos (facultative) and value. The chr column stand for the chromosome number, pos for the position of the nucleotide in its chromosome and value for the nucleosome occupancy. The data used to train the model are stored in traing_data.csv.gz

- **-o output_name:** the name of the .hdf5 file that will contain the model weights after training (register in Results_nucleosome). This name should match *weights_.+\.hdf5*

- **-model_name:** the name of the model we want to train. Could be cnn, cnn_deep, cnn_dilated, cnn_lstm.

- **-s:** if parsed indicates that the model we want to train or predict with is a seq2seq model.

- **-z:** if parsed indicates that we want to include the zeros in the training set (zeros are mainly non mappable regions, by default we do not take them in the training set as the value zeros is artificial).

- **-ds:** if training a seq2seq model downsampled the nucleosome density to decrease the length of the target sequence.

- **-t:** the chromsomes to include in the training set.

- **-v:** the chromosomes to include in the validation set.

### How to predict the nucleosome density on a new genome ?

- **Converting the fasta file:**

```NucleosomeDensity$ >>> python fasta_reader.py -d Programme/seq_chr_sacCer3/NewGenome -o ~/NucleosomeDensity/Programme/seq_chr_sacCer3/NewGenome```

- **-d:** the path to the directory where the fasta file are stored.

- **-o:** the path where we want the converted hdf5 file to be stored. (in Programme/seq_chr_sacCer3).

- **Predicting with the model:**

```NucleosomeDensity/Programme$ >>> python prediction.py -d NewGenome -w weights_CNN_nucleosome_in_vivo_all_data.hdf5 -m cnn --test 1```

- **-d:** the directory where the DNA sequence is stored in hdf5 (name of a subdirectory of seq_chr_sacCer3).

- **-w:** the name of the model to use for prediction (in Results_nucleosome). The available trained model is weights_CNN_nucleosome_in_vivo_all_data.hdf5.

- **--test:** the number of chromosome to predict on.

- **Predicting the single mutation effect:** 

```NucleosomeDensity/Programme$ >>> python mutazome.py -m weights_CNN_nucleosome_in_vivo_all_data.hdf5 -d sacCer3 -c 16```

- **-m:** name of the model to use to predict the effect of every single mutation.

- **-d:** directory where the DNA sequence is stored in hdf5 (name of a subdirectory of seq_chr_sacCer3).

- **-c:** chromosome on which to predict the single mutation effect.

### What are the data needed to begin ?

- the DNA sequence should be stored in hdf5 format with one file per chromosome in a directory set in *Programme/seq_chr*
- the nucleosome occupancy should be stored in a csv file (see previous section) in *Programme/Start_data*

### How could I find my results ?

- the weights of the trained model will be register in *Results_nucleosomes* with the name parsed as arguments (that should match *weights_.+\.hdf5*).
- the prediction are register in *Results_nucleosomes* with the name *y_pred_.+_applied_on_chromosomeN\.npy* (with *.+* behind the same as the one in the name of the file that contains the weights of the model).
