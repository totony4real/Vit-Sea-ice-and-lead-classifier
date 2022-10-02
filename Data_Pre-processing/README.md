# Data pre-processing

This folder contains all the code and the types of file you need for pre-precessing the data. The basic logic here is to find the co-located points between 
OLCI imagery and SRAL. And we label them and extract relevant patches from the original images. Now, let's have a look at the types of file you need for 
doing this. 
## Data_Pre-processing.ipynb
This is the main file used for processing. Open it in Jupyter Notebook.
## Two text files are used in the second cell in the code
```
with open(dir_path+'select_files_test.txt') as f:
    selects = [line.rstrip('\n') for line in open(dir_path+'select_files_test.txt')]
 with open(dir_path+'matching_SAR_tracks_test.txt') as f:
    matching_SAR_list = [line.rstrip('\n') for line in open(dir_path+'matching_SAR_tracks_test.txt')]
```
## matching_SAR_tracks_test.txt
A text file that contains the name of the corresponding SRAL files. In the example, there is only one (for the sake of convinences).
## matching_SAR_tracks_test.txt
A text file that contains the name of the corresponding OLCI Imagery file. In the example, there is only one.


The output data for training will be saved automatically.


