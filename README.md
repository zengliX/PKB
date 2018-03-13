# PKB usage instruction
This webpage provides an instruction for using the PKB (Pathway-based Kernel Boosting) model. PKB is designed to perform classification analysis with gene expression data. It incorporates gene pathway information as prior knowledge, and performs selection of informative pathways at the same time as building the predictive model. 

**Software requirement**:   
The program is written in **Python3**. Please install the following Python packages before running the program: 

- pandas
- numpy
- sharedmem
- scipy
- multiprocessing
- yaml
- matplotlib
- pickle

**Page contents:**

- [About PKB](#pkb)
- [Data preparation](#data)
- [Running PKB](#run)
- [Results interpretation](#results)

## <a name=pkb></a> About PKB
PKB is a boosting-based method for utilizing pathway information to better predict clinical outcomes. It constructs base learners from each pathway using kernel functions. In each boosting iteration, it identifies the optimal base learner and adds it to the prediction function.

The algorithm has two parts. The first part is calculating an optimal number of iterations using cross validation (CV). In this part, we split the training data into 3-folds, fit the boosting model, and monitor the classification error and loss function at each iteration. The iteration with minimum CV loss is used as iteration numbers.

In part two, we use the whole training data to fit the boosting model to the previously calculated number of iterations. We provide figures and tables to report the estimated weights for each pathway in the final model. If gene expression data for new samples is given, we also provide predictions in the output.

## Reference
_Zeng, L., Yu, Z. and Zhao, H. (2017) A pathway-based kernel boosting method for sample classification using genomic data_ [\[pdf\]](https://arxiv.org/pdf/1803.03910.pdf)

## <a name=data></a> Data preparation
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
test_file: test_predictor.txt

# model parameters
maxiter: 500
learning_rate: 0.02
Lambda:   
kernel: rbf
method: L1
```

The parameters are interpreted as following:

- `input_folder`: the folder where you keep the input data (path relative to your current folder)
- `output_folder`: the folder where you want to PKB to keep the output figures and data (path relative to input_folder)
- `predictor`: training data gene expression file (path relative to input_folder)
- `response`: training data clinical outcome file (path relative to input_folder)
- `predictor_set`: input pathway file (path relative to input_folder)
- `test_file`(optional): gene expression data for prediction; same format as `predictor`
- `maxiter`: number of maximum boosting iterations
- `learning_rate`: the learning rate parameter $\nu$ 
- `Lambda`(optional): the penalty parameter. If left blank, PKB will use an auto-determine algorithm to choose one.
- `kernel`: the kernel function. Currently we support, radial basis function(rbf) and polynomial kernel with $k$ degrees (poly2,poly3,etc)
- `method`: `L1` for $L_1$ penalty, `L2` for $L_2$ penalty

## <a name=run></a> Running PKB
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

## <a name=results></a> Results interpretation

### Figures
1. `CV_err.png, CV_loss.png`:    
present classifcation error and loss function value at each iteration of the cross validation process
![](example/example_output/CV_err.png?raw=true)
![](example/example_output/CV_loss.png?raw=true)


2. `opt_weights.png`:    
shows the estimated pathways weights fitted using our boosting model
![](example/example_output/opt_weights.png?raw=true)


3. `weights_path.png`:    
shows the changes of pathways' weights as iteration number increases.
![](example/example_output/weights_path.png?raw=true)


### Tables
1. `opt_weights.txt`:    
a table showing the optimal weights of all pahtways. It is sorted in descending order. The first column are pathways, and the second column are correponding weights.

2. `test_prediction.txt`:   
the predicted outcome values, if `test_file` is provided in the configuration file.

### Pickle file
1. `results.pckl`:   
contains information of the whole boosting process. You can recover the prediction function at every step from this file.


## Contact 
Please feel free to contact <li.zeng@yale.edu> if you have any questions.
