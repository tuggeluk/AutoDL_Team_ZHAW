import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    VectorizerMixin,
    _document_frequency,
)
from sklearn.feature_extraction._hashing import transform as htransform

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.sparse as sparse

from joblib import Parallel, delayed


def build_analyzer(analyzer='word', ngram_range=(1, 3),stop_words=None):
    v = VectorizerMixin()
    v.analyzer = analyzer
    v.preprocessor = None
    v.stop_words = None
    v.ngram_range = ngram_range
    v.strip_accents = False
    v.lowercase = True
    v.tokenizer = None
    v.token_pattern = r"(?u)\b\w\w+\b"
    v.input = 'content'
    v.stopwords = stop_words
    return v.build_analyzer()


def hash_vectorize(data, n_features, analyzer, ngram_range, stop_words):
    anal = build_analyzer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words)
    raw_X = (((token, 1) for token in anal(doc)) for doc in data)
    indices, indptr, values = htransform(
        raw_X,
        n_features,
        np.float64,
        True,
    )
    return indices, indptr, values


class ParallelHashingVectorizer(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            analyzer='word',
            ngram_range=(1, 3),
            n_features=2**20,
            n_jobs=4,
            stop_words=None
    ):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.stop_words = stop_words
        print("ParallelHashingVectorizer",self.stop_words,self.analyzer)

    def fit(self, X, y=None, **fit_params):
        return self

    def partial_fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        batch_size = math.ceil(len(X) / self.n_jobs)
        outs = Parallel(n_jobs=self.n_jobs)(
            delayed(hash_vectorize)(
                X[i:i+batch_size],
                self.n_features,
                self.analyzer,
                self.ngram_range,
                self.stop_words
            )
            for i in range(0, len(X), batch_size)
        )

        values = np.hstack([
            tup[2]
            for tup in outs
        ])
        indices = np.hstack([
            tup[0]
            for tup in outs
        ])
        indptr = np.hstack([
            tup[1][1:] + sum(outs[j][1][-1] for j in range(i))
            if i > 0 else tup[1]
            for i, tup in enumerate(outs)
        ])

        res = sparse.csr_matrix(
            (
                values,
                indices,
                indptr
            ),
            dtype=np.float64,
            shape=(len(X), self.n_features),
        )

        del values
        del indices
        del indptr

        res.sum_duplicates()
        return res

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class OnlineTfIdfTransformer(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            n_features=2**18,
    ):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.n_features = n_features

        self._n_samples = None
        self._df = None
        self._idf = None

        if self.use_idf:
            self.__init_idf()

    def __init_idf(self):
        added_docs = int(self.smooth_idf)
        self._n_samples = added_docs
        self._df = added_docs * np.ones(self.n_features)
        self._idf = np.ones(self.n_features)

    def partial_fit(self, X, y=None):
        # just assume X is sparse

        if self.use_idf:
            n_samples = X.shape[0]
            df = _document_frequency(X)

            self._n_samples += n_samples
            self._df += df
            self._idf = np.log(self._n_samples / self._df) + 1

        return self

    def fit(self, X, y=None):
        if self.use_idf:
            self.__init_idf()
        return self.partial_fit(X, y)

    def transform(self, X, y=None):
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            diag = sparse.diags(
                self._idf,
                offsets=0,
                shape=(self.n_features, self.n_features),
                format='csr',
            )
            X = X * diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class OnlineTfIdfVectorizer(HashingVectorizer):

    def __init__(
            self,
            input='content',
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
            analyzer='word',
            n_features=(2 ** 20),
            binary=False,
            norm='l2',
            alternate_sign=True,
            dtype=np.float64,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            n_features=n_features,
            binary=binary,
            norm=None,  # / ! \ we normalize in the idf transformer
            alternate_sign=alternate_sign,
            dtype=dtype,
        )

        self._idf = OnlineTfIdfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            n_features=n_features,
        )

    def partial_fit(self, X, y=None):
        X = super().partial_fit(X, y).transform(X)
        self._idf.partial_fit(X)
        return self

    def fit(self, X, y=None):
        X = super().fit(X, y).transform(X)
        self._idf.fit(X)
        return self

    def transform(self, X):
        X = super().transform(X)
        return self._idf.transform(X)

    def fit_transform(self, X, y=None):
        X = super().fit(X, y).transform(X)
        return self._idf.fit_transform(X)


def pipeline(n_grams, lang,  loss='log', penalty='l2',
             alpha=0.0001, l1_ratio=0.5, seed=0xDEADBEEF,  analyzer="char", stop_words=False):
    print('analyser', lang.lower() == 'en', analyzer)
    return Pipeline(
        memory=None,
        steps=[
            # ('vectorizer', OnlineTfIdfVectorizer(
            #     ngram_range=(1, 3),
            #     binary=False,
            #     analyzer='word' if lang.lower() == 'en' else 'char_wb',
            #     n_features=2 ** 20,
            #     norm='l2',
            # )),
            ('hasher', ParallelHashingVectorizer(
                analyzer='word' if lang.lower() == 'en' and analyzer != "char" else 'char_wb',
                ngram_range=n_grams,
                n_features=2 ** 20,
                n_jobs=4,
                stop_words='english' if lang.lower() == 'en' and stop_words else None,
            )),
            ('idf', OnlineTfIdfTransformer(
                norm='l2',
                n_features=2 ** 20,
            )),
            ('clf', SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10000,
                shuffle=False,
                n_jobs=4,
                random_state=seed,
                learning_rate='optimal',
                early_stopping=True,
                n_iter_no_change=10,
                class_weight='balanced',
                average=False,
            ))
        ]
    )


n_grams_zh = [
    (1, 1),
    (1, 2),
    (2, 3),
    (1, 5),
    (3, 5),
    (1, 7),
    (2, 4),
    (1, 9),
]
n_grams_en = [
    (1, 1, "word"),
    (1, 1, "char"),
    (1, 2, "word"),
    (2, 3, "word"),
    (2, 3, "char"),
    (1, 5, "word"),
    (3, 5, "word"),
    (1, 7, "word"),
]


class Model(object):

    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata

        self.seed = 0xDEADBEEF
        self.preds_c = None

        self.clf = pipeline(
            n_grams=(1, 1),
            lang=self.metadata["language"],
            seed=self.seed,
            analyzer="word" if self.metadata["language"] == "EN" else "char",
        )
        self.clfs = []
        self.n_samples = 4096
        self.steps = 0
        self.passes = 0
        if self.metadata["language"] == "EN":
            self.n_grams = n_grams_en
        else:
            self.n_grams = n_grams_zh
        print("language", self.metadata["language"])

    def train(self, train_dataset, remaining_time_budget=None):
        if self.done_training:
            return

        x_txt, y = train_dataset
        y = np.argmax(y, axis=1)
        self.passes += 1

        if self.n_samples >= len(x_txt):
            print("steps", self.steps)
            self.steps += 1
            self.clf = pipeline(
                n_grams=self.n_grams[self.steps][:2],
                lang=self.metadata["language"],
                seed=self.seed,
                analyzer=self.n_grams[self.steps][2] if self.metadata["language"] == "EN" else "char",
                stop_words=True if self.steps > 2 else False
            )
            x_train = x_txt
            y_train = y
            if self.steps == len(self.n_grams)-1:
                self.done_training = True
        else:
            self.steps = 1

            x_train, _, y_train, _ = train_test_split(
                x_txt, y,
                train_size=self.n_samples,
                random_state=self.seed,
                shuffle=True,
                stratify=y,
            )

            if self.passes == 1:
                x_train = [tk[:300] for tk in x_train]

        self.clf.fit(
            x_train,
            y_train,
        )
        if self.steps > 1:
            self.clfs.append(self.clf)
        else:
            self.clfs = [self.clf]

        self.n_samples *= 2

    def test(self, x_test, remaining_time_budget=None):
        if self.passes == 1:
            t1 = self.clfs[-1].predict_proba([tk[:300] for tk in x_test])
        else:
            t1 = self.clfs[-1].predict_proba(x_test)

        if self.preds_c is None:
            self.preds_c = t1
        else:
            self.preds_c += t1
        
        out = self.preds_c/len(self.clfs)        
        return out
