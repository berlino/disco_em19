# A Neural Two-Stage Approach for Recognizing Discontiguous Entities

## Setup

PyTorch (tested on 0.4), Python(v3)

## Code Structure

* module/SegGraph.py:  the segmental hypergraph for extracting segments
* model/Coarse2Fine.py: the joint model that performs the segment extraction and merging
* config.py: model and training configurations such as number of hidden units in LSTMs
* train.py: performs training and testing

## Data

We provide some processed sample data in the form of pkl files for demo. (data/examples.pkl and data/word\_vec\_200.pkl)

Note that we cannot distribute the data, the preprocessing scripts can be found from [Aldrian's code](http://www.statnlp.org/paper/learning-to-recognize-discontiguous-entities.html).

## Training and Testing

Simply run 'python train.py' will start the training process. We set the batch size to be 1 during training. After each epoch of training, the script outputs two kinds of metrics:

* precision/recall/f1 on the segment extraction
*  precision/recall/f1 on final entity extraction.

The model that performs best on the development set will be selected to be evaluated on the test set at the end of the script.


