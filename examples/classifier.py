import torch
import fitfast.trainer as ff
from fitfast.trainer import *
from fitfast.classifiers.linear import Linear
from fitfast.tricks.losses import OHEMLoss
from fitfast.tricks.regularizers import seq2seq_regularizer

CUDA_ID = 0
work_dir = './data/example'

def classifier():    
    itos = pickle.load(open(Path(work_dir) / 'tmp' / 'itos.pkl', 'rb'))
    vocab_size = len(itos)

    # instantiate the language model object
    lm = AWDLSTMModeler()

    # instantiate the classification model object
    clf = Linear()

    # call the model loader class
    loader = ff.ClassifierLoader(work_dir, n_tokens=vocab_size, bs=64, bptt=60, 
                                 sampler='weighted')

    # TO DO: IMRPOVE DROPOUT PARAMETRIZATION
    # FIX CALL ON EVAL/TEST
    dropouts = {'drop_i': .4, 
                'drop_e': .05,
                'drop_h': .2, 
                'drop_d': (.4, .1), 
                'w_drop': .4}

    # fetch the model/learner with the Loader object
    learner = loader.get_clf(lm, clf, **dropouts)
    
    # define the optimizer as a partial
    optimizer = partial(optim.Adam, betas=(.8, .99))
    regularizer = partial(seq2seq_regularizer, ar=2, tar=1)
    crit = OHEMLoss(.8)
    lparams = ff.LearningParameters(finetune=False, discriminative=True)
    
    # set learning parameters
    lparams.n_cycles = 25
    lparams.clip = 0.2
    lparams.lrs = 1e-2
    lparams.wds = 1e-6
    lparams.use_alt_clr = (10, 10, .95, .95)

    # compile the learner object
    learner.compile(work_dir, lparams, optimizer=optimizer, regularizer=regularizer,
                    metrics=[accuracy], crit=crit, 
                    callbacks=[TensorboardLogger(learner, work_dir, 
                                                 finetune=False)])
    
    # choose the unfreezing scedule
    learner.thaw('lm', gradual=True)

    # train learner
    learner.fit()
    
    # save the classifier
    learner.save('classifier')
    learner.sched.plot_lr()
    learner.sched.plot_loss()
    return True

if __name__ == '__main__': 

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Switching to CPU.')
        CUDA_ID = -1
    torch.cuda.set_device(CUDA_ID)

    classifier()

    # TO DO: replace wd with work_dir to avoid collision with weight decay