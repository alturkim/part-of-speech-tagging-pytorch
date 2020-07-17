# Character BiLSTM-CRFs Part-of-Speech Tagger
Coming: web interface for the model...

Part-of-speech-tagger using Character BiLSTM and Conditional Random Fields. 

# Dataset
Arabic PADT part of the Universal Dependency corpus.  
* Domain: mainly newswire  
* Sentences: 7,664 sentences  
* Tags: 17 UPOS tags  
For more details, see the [treebank page on UD website.](https://universaldependencies.org/treebanks/ar_padt/index.html)


# Model
## Architectures:
The model consists of two character-bidirectional-LSTM layers followed by a Conditional Random Fields Classifier.
## Optimizer:
Adam
## Hyperparameters
See `config.json`
# Results
F1 score on test set: 0.9241
 
# Usage
To train the model run the following command:  
`python train.py --data_dir DATA_DIR --config_dir CONFIG_DIR --checkpoint_dir CHECKPOINT_DIR --checkpoint_file CHECKPOINTFILE --maps_file MAP_FILE`
To evaluate the model on the test data run the following command:   
`python  evaluate.py --data_dir DATA_DIR --config_dir CONFIG_DIR --checkpoint_dir CHECKPOINT_DIR --checkpoint_file CHECKPOINTFILE --maps_file MAP_FILE`
### Note
map_file.pth.tar contains dictionaries used to maps characters and tags during the training process, it will be created 
before training, if not available, and should be used when evaluating the model on new datasets.
# Resources
[Stanford CS230 Pytorch code examples.](https://github.com/cs230-stanford/cs230-code-examples/tree/478e747b1c8bf57c6e2ce6b7ffd8068fe0287056/pytorch/nlp)

[Pytorch Sequence Labeling Tutorial.](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/tree/041f75a37497bd1b712a426b7d18631251ecd749)