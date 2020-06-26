#!/bin/bash

OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
chromosome=""
verbose=0

while getopts "gfc:" opt; do
    case "$opt" in
    g)
        git clone https://github.com/etirouthier/NucleosomeDensity.git
        cd NucleosomeDensity/
        ;;
    f)  
        python fasta_reader.py -d ./Programme/seq_chr_sacCer3/sacCer3 -o ~/NucleosomeDensity/Programme/seq_chr_sacCer3/sacCer3
        ;;
    c)  
        chromosome=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

cd Programme

if [ $chromosome = 1 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 2 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 2 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 3 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 4 5 6 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 4 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 5 6 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 5 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 6 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5

elif [ $chromosome = 6 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 7 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 7 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 8 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 8 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 9 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 9 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 10 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 10 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 11 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 11 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 12 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5
    
elif [ $chromosome = 12 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 13 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5   
    
elif [ $chromosome = 13 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 14 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5  

elif [ $chromosome = 14 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 15 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5  
    
elif [ $chromosome = 15 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 16
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5  

elif [ $chromosome = 16 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 15
    python mutazome.py -d sacCer3 -c $chromosome -m weights_CNN_nucleosome_in_vivo_all_data_exclude$chromosome.hdf5