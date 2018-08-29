BOS = 'xbos'    # beginning-of-sentence tag
FLD = 'xfld'    # data field tag

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").\
          replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").\
          replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"').\
          replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').\
          replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, is_train, n_lbls=None, lang='en'):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls + 1, len(df.columns)):
            texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer(lang=lang).proc_all_mp(partition_by_cores(texts), 
                                                              lang=lang)
    return tok, list(labels)


def get_all(df, is_train, n_lbls=None, lang='en'):
    tok, labels = [], []
        for i, r in enumerate(df):
            tok_, labels_ = get_texts(r, is_train=is_train, n_lbls=n_lbls, 
                                                            lang=lang)
            tok += tok_
            labels += labels_
        return tok, labels


def tok2id(dir_path, is_train, max_vocab=30000, min_freq=1, backwards=False):
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    tmp_path = p / 'tmp'
    assert tmp_path.exists(), f'Error: {tmp_path} does not exist.'

    if is_train:
        trn_tok = np.load(tmp_path / 'tok_trn.npy')
        val_tok = np.load(tmp_path / 'tok_val.npy')

        freq = Counter(p for o in trn_tok for p in o)
        print(freq.most_common(25))
        itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
        stoi = collections.defaultdict(lambda:0,
                                       {v:k for k,v in enumerate(itos)})
        print(len(itos))

        trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
        val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

        np.save(tmp_path / 'trn_ids.npy', trn_lm)
        np.save(tmp_path / 'val_ids.npy', val_lm)

        if backwards:
            trn_bwd_lm = np.array([list(reversed([stoi[o] for o in p])) \
                                   for p in trn_tok])
            val_bwd_lm = np.array([list(reversed([stoi[o] for o in p])) \
                                   for p in val_tok])

            np.save(tmp_path / 'trn_ids_bwd.npy', trn_bwd_lm)
            np.save(tmp_path / 'val_ids_bwd.npy', val_bwd_lm)

        pickle.dump(itos, open(tmp_path / 'itos.pkl', 'wb'))
    else:
        assert (Path(tmp_path / 'tok_trn.npy').exists(),
                f'Error: tok_trn.npy does not exist. '
                f'Use utils/process_data.py to create file.')

        trn_tok = np.load(tmp_path / 'tok_trn.npy')
        tst_tok = np.load(tmp_path / 'tok_tst.npy')
        freq = Counter(p for o in trn_tok for p in o)

        itos = [o for o,c in freq.most_common(max_vocab) if c > min_freq]
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
        stoi = collections.defaultdict(lambda:0,
                                       {v:k for k,v in enumerate(itos)})

        tst_lm = np.array([[stoi[o] for o in p] for p in tst_tok])
        np.save(tmp_path / 'tst_ids.npy', tst_lm)

        if backwards:
            tst_bwd_lm = np.array([list(reversed([stoi[o] for o in p])) \
                                   for p in tst_tok])
            np.save(tmp_path / 'tst_ids_bwd.npy', tst_bwd_lm)

    return


def save_toks(dir_path, save_as, tmp_path, df, is_train, n_lbls=None, lang='en'):
    if is_train:
        df = pd.read_csv(dir_path / f'{save_as}.csv', header=0)
        tok, lbl = get_all(df, is_train=is_train, n_lbls=n_lbls, lang=lang)
        np.save(tmp_path / f'tok_{save_as}.npy', tok)
        np.save(tmp_path / f'lbl_{save_as}.npy', lbl)

    else:
        df.to_csv(dir_path / f'{save_as}.csv', index=False, header=0)
        tok = get_all(df, is_train=is_train, n_lbls=n_lbls, lang=lang)
        np.save(tmp_path / f'tok_{save_as}.npy', tok)

    return tok


def create_toks(dir_path, is_train, file_name=None, n_lbls=None, lang='en', 
                backwards=False):
    
    print(f'dir_path {dir_path} file_name {file_name}'
          f'n_lbls {n_lbls} lang {lang} backwards {backwards}')

    try:
        spacy.load(lang)
    except OSError:
        # TODO handle tokenization of Chinese, Japanese, Korean
        print(f'spacy tokenization model is not installed for {lang}.')
        lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] \
                    else 'xx'
        print(f'Command: python -m spacy download {lang}')
        sys.exit(1)

    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'

    # create/check a temporary directory
    # for saving tokens and ids
    tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)

    # create train and val csv for fine-tuning if not present
    if is_train:
        # NOTE not using this check on test tokens
        if not (Path(dir_path / 'train.csv').exists()
        and Path(dir_path / 'val.csv').exists()):
            # load and shuffle data
            df = pd.read_csv(dir_path / file_name, index_col=False)
            df.label = pd.Categorical(df.label)
            df.label = df.label.cat.codes

            df = df.sample(frac=1).reset_index(drop=True)

            # train / validation split and save
            shuff = np.random.rand(len(df)) < 0.9
            trn = df[shuff]
            val = df[~shuff]

            trn.to_csv(dir_path / 'train.csv', index=False,
                          columns=['label', 'text'])
            val.to_csv(dir_path / 'val.csv', index=False,
                          columns=['label', 'text'])

        trn = pd.read_csv(dir_path / 'train.csv', header=0)
        val = pd.read_csv(dir_path / 'val.csv', header=0)

        tok_trn = save_toks(dir_path, 'train', tmp_path, df_trn,
                            is_train=is_train, n_lbls=n_lbls, lang=lang)
        save_toks(dir_path, 'val', tmp_path, df_val,
                  is_train=is_train, n_lbls=n_lbls, lang=lang)
        del trn, val

        trn_joined = [' '.join(o) for o in tok_trn]
        open(tmp_path / 'joined.txt', 'w',
            encoding='utf-8').writelines(trn_joined)

        tok2id(dir_path, is_train=is_train, max_vocab=30000,
               min_freq=1, backwards=backwards)
    else: pass
    return