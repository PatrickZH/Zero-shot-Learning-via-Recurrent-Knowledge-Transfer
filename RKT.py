import numpy as np
import scipy.io as sio
import sklearn.linear_model as linear
import copy
from tqdm import tqdm

def RKT_(emb, feat, ns, lambda_, gamma, locality=False, rkt=True):
    # parameters:
    # emb: [e^s, e^u]; n*de
    # feat: [f^s, f^u]; n*df
    # ns: number of seen classes
    # lambda_, gamma: sparsity and concatenation parameters, please refer to the paper

    # initialization
    n = feat.shape[0]
    D = np.zeros((n, n))
    alpha = np.zeros((n, n))
    lasso = linear.Lasso(lambda_)
    max_iter = 20

    if rkt == False:
        max_iter = 1

    if locality:
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i,j] = np.log(1+np.linalg.norm(emb[i]-emb[j], 2))
            D[i,:] = D[i,:]/(np.mean(D[i,:])+0.000001)
            D[i,i] = 1
        rD = 1 / (D[:, :] + 0.000001)
    else:
        D = np.identity(n)
        rD = D


    for it in range(max_iter):
        alpha_old = alpha
        alpha = np.zeros((n, n))

        # update alpha
        for i in range(n):
            if it>0:
                ef = np.concatenate((emb[i], gamma * feat[i]), axis=-1)
                EFD = np.matmul(rD, np.concatenate((emb, gamma*feat), axis=-1))
            else:
                ef = emb[i]
                EFD = np.matmul(rD, emb)
            EFD[i, :] = 0
            lasso.fit(X=EFD.transpose(), y=ef)
            alpha[i,:] = lasso.coef_

        # update F^u
        theta = np.identity(n) - alpha
        theta_s = theta[:, :ns]
        theta_u = theta[:, ns:]
        Fs = feat[:ns]
        Fu = - np.matmul(np.matmul(np.linalg.pinv(theta_u), theta_s), Fs)
        feat = np.concatenate((Fs, Fu), axis=0)

        # converge
        converge = np.sum((alpha - alpha_old)**2)
        print('train iter = %04d/%04d, update = %.4f'%(it, max_iter, converge))
        if converge < 0.0001 and it>3:
            print('converge')
            break

    return alpha, Fu


''' data processing '''
# put data into folder '../data/'

dataset = 'AwA'

data = sio.loadmat('../data/%s_ImageFeatures_VGG'%dataset)
ImageFeatures = data['ImageFeatures']
Labels = data['Labels'].reshape(-1).tolist()

data = sio.loadmat('../data/%s_WordVectors'%dataset)
WordVectors = data['WordVectors']

data = sio.loadmat('../data/%s_Attributes'%dataset)
Attributes = data['attributes_embedding_c']

data = sio.loadmat('../data/%s_splits_default'%dataset)
splits = data['splits'].reshape(-1).tolist()

Labels = [lab-1 for lab in Labels]
splits = [lab-1 for lab in splits]

''' train '''
cls = len(list(set(Labels)))
unseen = splits
seen = [i for i in range(cls) if i not in unseen]
ns = len(seen)
nu = len(unseen)

emb = WordVectors
emb = np.concatenate((emb[seen], emb[unseen]), axis=0)

feat = np.zeros((cls, ImageFeatures.shape[1]))
for i in range(cls):
    idx = [id for id, lab in enumerate(Labels) if lab==i]
    feat[i] = np.mean(ImageFeatures[idx], axis=0)
feat = np.concatenate((feat[seen], np.random.randn(nu, ImageFeatures.shape[1])), axis=0)

lambda_ = 0.01
gamma = 0.01
locality = False # only for large-scale datasets (e.g. ImageNet)
rkt = True # True: ours, False: baseline

alpha, Fu = RKT_(emb, feat, ns, lambda_, gamma, locality=locality, rkt=rkt)

''' classification '''
idx_unseen = [i for i, lab in enumerate(Labels) if lab in unseen]
Labels_unseen = [Labels[idx] for idx in idx_unseen]
num_unseen = len(idx_unseen)

# nearest classification
acc = 0
Labels_predict = []
for idx in tqdm(idx_unseen):
    f = ImageFeatures[idx].reshape(-1)
    dis = np.sqrt(np.sum((np.tile(f[np.newaxis, :], (nu, 1)) - Fu)**2, axis=-1))
    lab = unseen[np.argmin(dis, axis=-1)]
    Labels_predict.append(lab)
acc = np.sum(np.asarray(Labels_predict, dtype=np.int)==np.asarray(Labels_unseen, dtype=np.int))/num_unseen
print('----------------------------')
print('lambda = %.4f, gamma = %.4f, locality = %s, rkt = %s \naccuracy = %.4f' % (lambda_, gamma, locality, rkt, acc))





