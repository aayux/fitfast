import torch
import fitfast as ff
from fitfast.lm import AWDLSTMModeler
from fitfast.utils.data import load_csv, train_split

def main():
    corpus = load_csv('data.csv', lang='en')
    train, val = train_split(corpus)
    
    lm = AWDLSTMModeler()
    loader = ff.Loader(train, val)
    learner = loader.get_model(lm)
    
    learner.compile(clip=0.2,
                    metrics=[accuracy])
    
    callbacks = [TensorboardLogger()]
    
    learner.fit(callbacks=callbacks)
    learner.save('example')
    learner.save_encoder('example_encoder')

if __name__ == '__main__': main()