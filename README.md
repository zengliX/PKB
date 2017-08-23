# PKB usage instruction
This webpage provides an instruction for using the PKB (Pathway-based Kernel Boosting) model. PKB is designed to perform classification analysis with gene expression data. It incorporates gene pathway information as prior knowledge, and performs selection of informative pathways at the same time as building the predictive model. 

**Software preparation**:   
The program is written in **Python3**. Please install the following Python packages before running the program: 

- pandas
- numpy
- sharedmem
- scipy
- multiprocessing
- yaml
- matplotlib
- pickle

## Data preparation
PKB requires the input datasets to be formatted in certain ways. 

### Clinical outcome input
Please refer to `example/response.txt` for an example. It needs to be a `,`-separated file with two columns, one for sample ID and the other for outcome value. The first row should be column names. The `response` column is `-1,1` coded, indicating different sample classes. 

Example:

  sample | response 
  ------- | --------- 
  sample1 | 1 
  sample2 | 1
  sample3 | -1
  sample4 | -1   
  ...     | ... 


### Gene expression input
Please refer to `example/predictor.txt` for an example. It is also a comma-separated file. The first column is sample ID, and the other columns are genes. The first row is columns names, and each other row represents one sample.

Example:

| sample  | gene1 | gene2 | gene3 | gene4 | ... |
|---------|-------|-------|-------|-------|-----|
| sample1 | 1.2   | 3.3   | 4.5   | 0.1   | ... |
| sample2 | 0.5   | 2.6   | 2.3   | 1.2   | ... |
| sample3 | 0.1   | 1.4   | 0.1   | 2.2   | ... |
| sample4 | 0.8   | 0.2   | 8.6   | 1.8   | ... |
| ...     | ...   | ...   | ...   | ...   | ... |

### Pathway input
You can either provide your own pathway file, or use the built-in files, including  **KEGG, Biocarta, GO biological process pathways, GO computional pathways**. 

To use the built-in pathways, just use the corresponding files in `./data` folder when writing the configuration file. 

If you would like to use customized pathway file, please refer to `example/predictor_sets.txt` for an example. It should be a comma-separated file with no header. The first column are the names of pathways, and the second column are the lists of individual pathway members. Each list is a string of genes separated by spaces.

Example:

  pathway| contents 
  ------- | --------- 
  pathway1 | gene11 gene12 gene13 gene14 
  pathway2 | gene21 gene22
  pathway3 | gene31 gene32 gene33
  pathway4 | gene41 gene42
  ...     | ... 

### PKB configuration file
Here is an example configuration file for applying PKB to our example dataset:

```python
# folders
input_folder: ./example
output_folder: example_output

# input files
predictor: predictor.txt  
response: response.txt    
predictor_set: predictor_sets.txt 

# model parameters
maxiter: 500
random_subset: 1
learning_rate: 0.02
Lambda:   
kernel: rbf
method: L1
test_ratio: 0.33 
```

The parameters are interpreted as following:

- `input_folder`: the folder where you keep the input data (path relative to your current folder)
- `output_folder`: the folder where you want to PKB to keep the output figures and data (path relative to `input_folder`)
- `predictor`: input gene expression file (path relative to `input_folder`)
- `response`: input clinical outcome file (path relative to `input_folder`)
- `predictor_set`: input pathway file (path relative to `input_folder`)
- `maxiter`: number of maximum boosting iterations
- `random_subset`: a value in `[0,1]`. Each time choose a random subset of samples to train the model. Choose a value `< 1` to speed up the boosting process when the dataset is large 
- `learning_rate`: the learning rate parameter $\nu$ 
- `Lambda`: the penalty parameter. If left blank, PKB will use an auto-determine algorithm to choose one.
- `kernel`: the kernel function. Currently we support, radial basis function(`rbf`) and polynomial kernel with $k$ degrees (`poly2`,`poly3`,etc)
- `method`: `L1` for $L_1$ penalty, `L2` for $L_2$ penalty
- `test_ratio`: the proportion of data to be used as testing dataset


## Running PKB
Follow the steps below in order to run PKB on your own computer (we use our toy dataset as example):

1. clone this git repository :

	```bash
	git clone https://github.com/zengliX/PKB PKB
	cd PKB
	```
2. prepare datasets and configuration files following the format given in the previous section

3. implement PKB: 

	```python
	# python PKB.py path/to/your_config_file.txt
	python PKB.py ./example/config_file.txt
	```

The outputs will be saved in the `output_folder` as you specified in the configuration file.

## Results interpretation


### Figures
1. `err.pdf`:    
presents the classifcation error rate at each iteration
![](./example/example_output/err.pdf)

2. `opt_weights.pdf`:    
shows the optimal pathways weights at the iteration with the smallest testing error
![](./example/example_output/opt_weights.pdf)


3. `weights_path1.pdf,weights_path2.pdf`:    
both figures show the changes of the pathways weights as iteration number goes up. `weights_path2.pdf` is just a zoomed-in version of `weights_path1.pdf`, zooming close to the optimal iteration.
![](./example/example_output/weights_path1.pdf)
![](./example/example_output/weights_path2.pdf)


### Table
1. `opt_weights.txt`:    
a table showing the optimal weights of all pahtways. It is sorted in descending order. The first column are pathways, and the second column are correponding weights.

### Pickle file
1. `results.pckl`:   
contains information of the whole boosting process. You can recover the prediction function at every step from this file.


## Reference
_Zeng, L., Yu, Z. and Zhao, H. (2017) A pathway-based kernel boosting model for classification analysis of genomic data_

## Contact 
Please feel free to contact <li.zeng@yale.edu> if you have any questions.