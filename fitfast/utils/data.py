from ..imports import *
import html

BOS = 'xbos'    # beginning-of-sentence tag
FLD = 'xfld'    # data field tag
re_ = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").\
          replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").\
          replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"').\
          replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').\
          replace('\\', ' \\ ')
    return re_.sub(' ', html.unescape(x))


class Preprocess(object):
    def __init__(self, lang='en'):
        self.wd = None
        self.tmp = None
        self.file = None
        self.itos = None
        self.lang = lang

        try:
            spacy.load(self.lang)
        except OSError:
            print(f'spacy tokenization model is not installed for {self.lang}.')
            self.lang = self.lang if self.lang in \
                        ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
            print(f'Command: python -m spacy download {self.lang}')
            sys.exit(1)

    def _make_token(self, df):
        if len(df.columns) == 1:
            labels = []
            texts = f'\n{BOS} {FLD} 0 ' + df[0].astype(str)
        else:
            labels = df[0].values.astype(np.int64)
            texts = f'\n{BOS} {FLD} 0 ' + df[1].astype(str)
            # for i in range(n_lbls + 1, len(df.columns)):
            #     texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
        texts = list(texts.apply(fixup).values)
        tokens = Tokenizer(lang=self.lang).proc_all_mp(partition_by_cores(texts), 
                                                                 lang=self.lang)
        return tokens, list(labels)


    def _get_tokens(self, df):
        tokens, labels = [], []
        for i, r in enumerate(df):
            tokens_, labels_ = self._make_token(r)
            tokens += tokens_
            labels += labels_
            return tokens, labels


    def tok2id(self, max_vocab=30000, min_freq=1):
        train_tok = np.load(self.tmp / 'tok_trn.npy')
        val_tok = np.load(self.tmp / 'tok_val.npy')

        freq = Counter(p for o in train_tok for p in o)
        print(freq.most_common(25))
        self.itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
        self.itos.insert(0, '_pad_')
        self.itos.insert(0, '_unk_')
        stoi = collections.defaultdict(lambda: 0, 
                                       {v:k for k,v in enumerate(itos)})

        train_ids = np.array([[stoi[o] for o in p] for p in train_tok])
        val_ids = np.array([[stoi[o] for o in p] for p in val_tok])

        np.save(self.tmp / 'train_ids.npy', train_ids)
        np.save(self.tmp / 'val_ids.npy', val_ids)

        # if backwards:
        #     train_bwd_ids = np.array([list(reversed([stoi[o] for o in p])) \
        #                               for p in train_tok])
        #     val_bwd_ids = np.array([list(reversed([stoi[o] for o in p])) \
        #                             for p in val_tok])
        #     np.save(tmp_path / 'train_ids_bwd.npy', train_bwd_ids)
        #     np.save(tmp_path / 'val_ids_bwd.npy', val_bwd_ids)

        pickle.dump(self.itos, open(self.tmp / 'itos.pkl', 'wb'))
        return train_ids, val_ids

    def tokenize(self, file):
        df = pd.read_csv(self.wd / f'{file}.csv', header=0)
        tokens, labels = self._get_tokens(df)
        np.save(self.tmp / f'tok_{file}.npy', tokens)
        if len(df.columns) > 1: 
            np.save(self.tmp / f'lbl_{file}.npy', labels)
        return tokens


    def load(self, wd, file, split=.9):
        self.wd = Path(wd)
        self.file = wd / file
        assert self.file.exists(), f'Error: {self.file} does not exist.'

        # create/check a temporary directory
        self.tmp = self.wd / 'tmp'
        self.tmp.mkdir(exist_ok=True)

        df = pd.read_csv(self.file, index_col=False)
        
        if len(df.columns) > 1:
            df.label = pd.Categorical(df.label)
            df.label = df.label.cat.codes

        train, val = self.split_train_val(df, split)

        if len(df.columns) > 1:
            train.to_csv(self.wd / 'train.csv', index=False, 
                         columns=['label', 'text'])
            val.to_csv(self.wd / 'val.csv', index=False, 
                         columns=['label', 'text'])
        else:
            train.to_csv(self.wd / 'train.csv', index=False, columns=['text'])
            val.to_csv(self.wd / 'val.csv', index=False, columns=['text'])
        
        del train, val, df

        tokens = self.tokenize('train')
        self.tokenize('val')

        joined = [' '.join(o) for o in tokens]
        open(tmp / 'joined.txt', 'w', encoding='utf-8').writelines(joined)

        return self.tok2id(max_vocab=30000, min_freq=1)

        def vocabulary(self):
            return self.itos

        def split_train_val(self, df, split=.9):
            df = df.sample(frac=1).reset_index(drop=True)
            shuff = np.random.rand(len(df)) < split
            return df[shuff], df[~shuff]