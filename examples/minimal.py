import torch
import fitfast.trainer as ff
from fitfast.utils.preprocessing import Preprocess
from fitfast.trainer import *
from fitfast.lm import AWDLSTMModeler
from fitfast.classifiers.linear import Linear

CUDA_ID = 0
work_dir = './data/example'

def main():
    # initialise the preprocessing object
    pp = Preprocess(lang='en')

    train, val = pp.load(work_dir, 'data.csv')
    itos, vocab_size = pp.vocabulary()

    # instantiate the language model object
    lm = AWDLSTMModeler()

    # call the model loader class
    loader = ff.LanguageModelLoader(train, val, lang='en')

    # fetch the model/learner with the Loader object
    # TO DO: Implement an abstract base class Loader
    learner = loader.get_model(lm, itos, finetune=True)
    lparams = ff.LearningParameters()

    # compile the learner object
    learner.compile(lparams, metrics=[accuracy], 
                    callbacks=[TensorboardLogger(learner, work_dir)])
    # train learner
    learner.fit()

    # save the language model and encoder
    learner.save('lm')
    learner.save_encoder('lm_encoder')

    # instantiate the classification model object
    clf = Linear()

    # call the model loader class
    loader = ff.ClassifierLoader(work_dir)

    # fetch the model/learner with the Loader object
    learner = loader.get_clf(lm, clf)

    lparams = ff.LearningParameters(finetune=False)

    # compile the learner object
    learner.compile(work_dir, lparams, metrics=[accuracy],
                    callbacks=[TensorboardLogger(learner, work_dir, 
                                                 finetune=False)])

    # choose the unfreezing scedule
    learner.thaw('lm')

    # train learner
    learner.fit()
    
    # save the classifier
    learner.save('classifier')
    
    # save pyplot curves
    learner.sched.plot_lr()
    learner.sched.plot_loss()
    
    # prepare test data
    test = pp.load(work_dir, 'test.csv', is_test)
    
    # call loader with test data
    # TO DO: make this seamless regardless of train or test
    loader = ff.ClassifierLoader(work_dir, is_test)
    
    # recall learner object
    learner = loader.get_clf(lm, clf)
    
    # get predictions
    preds = learner.predict()
    
    # display the test statistics
    ff.metrics(preds)

    return True

if __name == '__main__':
    
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Switching to CPU.')
        CUDA_ID = -1
    torch.cuda.set_device(CUDA_ID)
    
    main()
