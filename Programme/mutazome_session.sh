#!/bin/bash

if [ $1 = 1 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 2 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 2 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 2 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 3 4 5 6 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 3 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 4 5 6 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 4 5 6 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 4 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 5 6 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 5 6 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 5 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 6 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 6 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5

elif [ $1 = 6 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 7 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 7 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 7 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 8 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 8 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 8 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 9 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 9 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 9 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 10 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 10 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 10 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 11 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 11 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 11 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 12 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 12 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5
    
elif [ $1 = 12 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 13 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 13 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5   
    
elif [ $1 = 13 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 14 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 14 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5  

elif [ $1 = 14 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 15 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 15 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5  
    
elif [ $1 = 15 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 16

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 16 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5  

elif [ $1 = 16 ]
then

    python training.py -d sacCer3 -f proba_in_vivo.csv -o weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 15

    python prediction.py -d sacCer3 -f proba_in_vivo.csv -w weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5 -m cnn -t 1 2 3 4 5 6 7 8 9 10 11 12 13 -v 14 15 --test $1

    python mutazome.py -d sacCer3 -c $1 -m weights_CNN_nucleosome_in_vivo_all_data_exclude$1.hdf5