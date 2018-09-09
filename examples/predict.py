import torch
import fitfast.trainer as ff
from fitfast.utils.preprocessing import Preprocess
from fitfast.trainer import *
from fitfast.lm import AWDLSTMModeler
from fitfast.classifiers.linear import Linear

CUDA_ID = 0
work_dir = './data/example'
is_test = True

def main():
    # initialise the preprocessing object
    pp = Preprocess(lang='en')
    
    # prepare test data
    test = pp.load(work_dir, 'test.csv', is_test)
    _, vocab_size = pp.vocabulary(is_test)
    
    # instantiate the language model object
    lm = AWDLSTMModeler()

    # instantiate the classification model object
    clf = Linear()

    # call the model loader class
    loader = ff.ClassifierLoader(work_dir, is_test)
    
    # fetch the model/learner with the Loader object
    learner = loader.get_clf(lm, clf)
    
    # get predictions
    probs, preds = learner.predict(softmax=True)
        
    # display the test statistics
    ff.metrics(preds)

if __name == '__main__':

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Switching to CPU.')
        CUDA_ID = -1
    torch.cuda.set_device(CUDA_ID)

    main()