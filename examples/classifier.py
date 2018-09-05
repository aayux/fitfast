import torch
import fitfast.trainer as ff
from fitfast.trainer import *
from fitfast.classifiers.linear import Linear
from fitfast.tricks.lossess import NLLHardExampleLoss
from fitfast.tricks.regularizers import seq2seq_regularizer

CUDA_ID = 0

def classifier():
    work_dir = './data/example'
    
    # instantiate the classification model object
    clf = Linear()

    # call the model loader class
    loader = ff.ClassifierLoader(work_dir, bs=64, bptt=60, dims=(), 
                                 sampler=WeightedRandomSampler)

    # TO DO: IMRPOVE DROPOUT PARAMETRIZATION
    # FIX CALL ON EVAL/TEST
    dropouts = {'drop_i': .6, 
                'drop_e': .1, 
                'drop_h': .2, 
                'drop_d': .5, 
                'w_drop': .4}

    # fetch the model/learner with the Loader object
    learner = loader.get_clf(lm, clf, **dropouts)
    
    # define the optimizer as a partial
    optimizer = partial(optim.Adam, betas=(.8, .99))
    regularizer = partial(seq2seq_regularizer, ar=2, tar=1)
    crit = NLLHardExampleLoss(.8)
    lparams = ff.LearningParameters(discriminative=True)
    
    # set learning parameters
    lparams.n_cycles = 4
    lparams.clip = 0.2
    lparams.lrs = 1e-1
    lparams.wds = 1e-6
    lparams.use_alt_clr = (10, 10, .95, .95)

    # compile the learner object
    learner.compile(lparams, optimizer=optimizer, regularizer=regularizer,
                    metrics=[accuracy], crit=crit, 
                    callbacks=[TensorboardLogger(work_dir)])
    
    # choose the unfreezing scedule
    learner.unfreeze(gradual=True)

    # train learner
    learner.fit(save_best_model=True, wd=work_dir)
    
    # save the classifier
    learner.save(work_dir, 'classifier')
    return True

if __name__ == '__main__': 

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Switching to CPU.')
        CUDA_ID = -1
    torch.cuda.set_device(CUDA_ID)

    classifier()

    # TO DO: replace wd with work_dir to avoid collision with weight decay