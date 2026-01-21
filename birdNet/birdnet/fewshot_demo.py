import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_features(features_dir):
    meta_path = os.path.join(features_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        # try glob
        files = sorted([p for p in os.listdir(features_dir) if p.endswith('.npy')])
        meta = []
        for f in files:
            wav = f.replace('.npy', '.wav')
            meta.append((wav, os.path.join(features_dir, f)))
    else:
        meta = joblib.load(meta_path)
    X, y = [], []
    for wav_path, feat_path in meta:
        feat = np.load(feat_path)
        X.append(feat.flatten())
        # infer label from parent folder name if possible
        if wav_path and os.path.exists(wav_path):
            label = os.path.basename(os.path.dirname(wav_path))
        else:
            # try to parse from filename (prefix label_...)
            base = os.path.basename(feat_path)
            label = base.split('_')[0]
        y.append(label)
    return np.vstack(X), np.array(y)


def sample_fewshot(X, y, n_shots=5, n_queries=10, rng=0):
    import numpy as _np
    _np.random.seed(rng)
    labels = np.unique(y)
    support_X, support_y, query_X, query_y = [], [], [], []
    for lab in labels:
        ids = np.where(y==lab)[0]
        if len(ids) < n_shots + n_queries:
            continue
        perm = _np.random.permutation(ids)
        support_X.append(X[perm[:n_shots]])
        support_y += [lab]*n_shots
        query_X.append(X[perm[n_shots:n_shots+n_queries]])
        query_y += [lab]*n_queries
    if len(support_y)==0:
        raise RuntimeError('Not enough data per class for shots/queries')
    return np.vstack(support_X), np.array(support_y), np.vstack(query_X), np.array(query_y)


def main(features_dir, shots=5, queries=10):
    X, y = load_features(features_dir)
    sX, sy, qX, qy = sample_fewshot(X, y, n_shots=shots, n_queries=queries)
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(sX, sy)
    pred_knn = knn.predict(qX)
    print('k-NN accuracy', accuracy_score(qy, pred_knn))
    lr = LogisticRegression(max_iter=500).fit(sX, sy)
    pred_lr = lr.predict(qX)
    print('LogReg accuracy', accuracy_score(qy, pred_lr))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--features-dir', required=True)
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--queries', type=int, default=10)
    args = p.parse_args()
    main(args.features_dir, shots=args.shots, queries=args.queries)
# birdnet/fewshot_demo.py
import argparse
import numpy as np
import joblib
from glob import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

def load_features(features_dir):
    meta = joblib.load(os.path.join(features_dir, 'meta.pkl'))
    X, y = [], []
    for wav_path, feat_path in meta:
        feat = np.load(feat_path)
        X.append(feat.flatten())
        # infer label from parent folder name
        label = os.path.basename(os.path.dirname(wav_path))
        y.append(label)
    return np.vstack(X), np.array(y)

def sample_fewshot(X, y, n_shots=5, n_queries=10, rng=0):
    import numpy as _np
    _np.random.seed(rng)
    labels = np.unique(y)
    support_X, support_y, query_X, query_y = [], [], [], []
    for lab in labels:
        ids = np.where(y==lab)[0]
        if len(ids) < n_shots + n_queries:
            continue
        perm = _np.random.permutation(ids)
        support_X.append(X[perm[:n_shots]])
        support_y += [lab]*n_shots
        query_X.append(X[perm[n_shots:n_shots+n_queries]])
        query_y += [lab]*n_queries
    if len(support_y)==0:
        raise RuntimeError("Not enough data per class for shots/queries")
    return np.vstack(support_X), np.array(support_y), np.vstack(query_X), np.array(query_y)

def main(args):
    X, y = load_features(args.features_dir)
    sX, sy, qX, qy = sample_fewshot(X, y, n_shots=args.shots, n_queries=args.queries)
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(sX, sy)
    pred_knn = knn.predict(qX)
    print("k-NN accuracy", accuracy_score(qy, pred_knn))
    lr = LogisticRegression(max_iter=500).fit(sX, sy)
    pred_lr = lr.predict(qX)
    print("LogReg accuracy", accuracy_score(qy, pred_lr))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--features-dir', required=True)
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--queries', type=int, default=10)
    args = p.parse_args()
    main(args)