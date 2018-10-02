# KG-MIML-Net
Knowledge Guided Multi-instance Multi-label Learning via Neural Networks in Medicines Prediction

Paper has been accepted in [ACML, 2018](http://www.acml-conf.org/2018/) and paper link will be released later.
## Overview
This repository contains code necessary to run KG-MIML-Net model. KG-MIML-Net is mainly based on RNN encoder-decoder, and two additional knowledge, i.e., structural knowledge and contextual knowledge are combined to improve performance on medicines prediction task. KG-MIML-Net is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/) containing more than 40,000 patients and demonstrates its effectiveness compared with several state-of-the-art methods in MIML learning and heathcare area.

## Requirements
- Pytorch >=0.4
- Python >=3.5

## Running the code
### Data preprocessing
It's the toughest thing to prepare the fuel for the model. Fortunately, you can find the preprocessing code called **create_data.py** in ./data. But some data is missing because of the regulation of MIMIC-III like DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv. You should complete the required training course of MIMIC-III, then download and put the above mentioned files in ./data. For contextual knowledge, there are some codes in [mimic-code](https://github.com/MIT-LCP/mimic-code/tree/master/concepts) where demographics and measurements within firstday (e.g. age, weight, blood-gas) can be extracted. Then you may put it under ./data dir to use the contextual knowledge.

### Model Comparation
- K-most frequent: The simple baseline choose K medicines for a disease which is the most common K medicines co-occur with that disease. K is the hyper parameter determined according to the performance in evaluation set. **refer to ./baseline/k_most_frequency.py**
- MLP: The diseases and medicines are firstly transformed to multi-hot vector, the a 3-layer MLP is carried to make multi-label prediction. A global threshold is used to select positive medicines. **refer to ./baseline/mlp.py**
- MIMLFast: The traditional state-of-art MIML method transfered instances to label specific space and considered sub-concepts for each label. The representation of instances should be given first, so we use the skip-gram method to pre-generate the instances' representation. **refer to [matlab code](http://lamda.nju.edu.cn/code\_MIMLfast.ashx)**
- Leap: **refer to [neozhangthe1's Github](https://github.com/neozhangthe1/AutoPrescribe)**

### Model variants
To test the effectiveness of three different components proposed in KG-MIML-Net, the user can modify the config_template.json file.
```python
"add_contextual_layer": true # set true to use contextual knowledge
"add_supervised": true # set true to use supervised attention module
"add_tree_inputs_embedding": true # set true to use structural knowledge for input instances
"add_tree_outputs_embedding": true # set true to use structural knowledge for output instances
```

### Train and test
You can feel free to change the hyper-parameters and specify a model_name by changing "model_name" attr in in config_template.json file.
The ./main.py is the entrance for training, testing and sampling.
```python
python main.py # training script
python main.py -e # testing script
python main.py -p patient_id # sampling a patient 
```

### Acknowledgements
Pytorch framework is modified from [victoresque's github](https://github.com/victoresque/pytorch-template)
