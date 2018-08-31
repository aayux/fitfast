import torch

import fitfast as ff
from fitfast.lm import LSTMModeler
from fitfast.data_utils import load_csv, train_split

CUDA_ID = 0

def main():
    work_dir = './data/example'
    corpus, itos = load_csv(work_dir, 'data.csv', lang='en')
    train, val = train_split(corpus)
    
    # instantiate the language model object
    lm = LSTMModeler()

    # call the model loader class
    loader = ff.Loader(train, val, lang=lang, bs=64, bptt=60)
    
    # input dropout
    drop_i = .6
    
    # embedding dropout
    drop_e = .1
    
    # activation dropout
    drop_h = .2
    
    # decoder dropout
    drop_d = .5
    
    # weight dropout
    w_drop = .4

    # TO DO: IMRPOVE DROPOUT PARAMETRIZATION
    # FIX CALL ON EVAL/TEST

    # fetch the model/learner with the Loader object
    learner = loader.get_model(lm, itos, finetune=True)
    
    # define the optimizer as a partial
    optimizer = partial(optim.Adam, betas=(0.8, 0.99))
    regularizer = partial(seq2seq_regularizer, ar=2, tar=1)
    lparams = ff.LearningParameters(discriminative=True)
    
    # set learning parameters
    lparams.n_cycles = 12
    lparams.clip = 0.2
    lparams.lrs = 1e-1
    lparams.wds = 1e-6
    lparams.use_clr_alt = (10, 10, .95, .95)

    # compile the learner object
    learner.compile(lparams, optimizer=optimizer, regularizer=regularizer,
                    metrics=[accuracy], 
                    callbacks=[TensorboardLogger(), 
                               EarlyStopping(learner, work_dir, 'example')])
    # train learner
    learner.fit(save_best_model=True)
    
    # save the language model and encoder
    learner.save(work_dir, 'example')
    learner.save_encoder(work_dir, 'example_encoder')

if __name__ == '__main__': 

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Switching to CPU.')
        CUDA_ID = -1
    torch.cuda.set_device(CUDA_ID)

    main()