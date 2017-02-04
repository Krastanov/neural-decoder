from codes import ToricCode
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l1, activity_l1
from keras.optimizers import Nadam
from keras.objectives import binary_crossentropy


F = lambda _: K.cast(_, 'float32') # TODO XXX there must be a better way to calculate mean than this cast-first approach


class CodeCosts:
    def __init__(self, L, code, Z, X):
        self.L = L
        code = code(L)
        H = code.H(Z,X)
        E = code.E(Z,X)
        self.H = K.variable(value=H) # TODO should be sparse
        self.E = K.variable(value=E) # TODO should be sparse
    def exact_reversal(self, y_true, y_pred):
        "Fraction exactly predicted qubit flips."
        return K.mean(F(K.all(K.equal(y_true, K.round(y_pred)), axis=-1)))
    def non_triv_stab_expanded(self, y_true, y_pred):
        "Whether the stabilizer after correction is not trivial."
        return K.any(K.batch_dot(self.H,(K.round(y_pred)+y_true)%2, axes=[1,1])%2, axis=0)
    def logic_error_expanded(self, y_true, y_pred):
        "Whether there is a logical error after correction."
        return K.any(K.batch_dot(self.E,(K.round(y_pred)+y_true)%2, axes=[1,1])%2, axis=0)
    def triv_stab(self, y_true, y_pred):
        "Fraction trivial stabilizer after corrections."
        return 1-K.mean(F(self.non_triv_stab_expanded(y_true, y_pred)))
    def no_error(self, y_true, y_pred):
        "Fraction no logical errors after corrections."
        return 1-K.mean(F(self.logic_error_expanded(y_true, y_pred)))
    def triv_no_error(self, y_true, y_pred):
        "Fraction with trivial stabilizer and no error."
        # TODO XXX Those casts (the F function) should not be there! This should be logical operations
        triv_stab = 1 - F(self.non_triv_stab_expanded(y_true, y_pred))
        no_err    = 1 - F(self.logic_error_expanded(y_true, y_pred))
        return K.mean(no_err*triv_stab)

def create_model(L, hidden_sizes=[4], hidden_act='tanh', act='sigmoid', loss='binary_crossentropy',
                 Z=True, X=False, learning_rate=0.002):
    in_dim = L**2 * (X+Z)
    out_dim = 2*L**2 * (X+Z)
    model = Sequential()
    model.add(Dense(int(hidden_sizes[0]*out_dim), input_dim=in_dim, init='glorot_uniform'))
    model.add(Activation(hidden_act))
    for s in hidden_sizes[1:]:
        model.add(Dense(int(s*out_dim), init='glorot_uniform'))
        model.add(Activation(hidden_act))
    model.add(Dense(out_dim, init='glorot_uniform'))
    model.add(Activation(act))
    c = CodeCosts(L, ToricCode, Z, X)
    model.compile(loss=loss,
                  optimizer=Nadam(lr=learning_rate),
                  metrics=[c.exact_reversal, c.triv_stab, c.no_error, c.triv_no_error]
                 )
    return model

def data_generator(code, L, p, batch_size=512, Z=True, X=False):
    H = []
    code = code(L)
    in_dim = L**2 * (X+Z)
    out_dim = 2*L**2 * (X+Z)
    H = code.H(Z,X)
    Ht = H.T
    while True:
        flips = np.zeros((batch_size, out_dim), dtype=np.dtype('b'))
        rand = np.random.rand(batch_size, 2*L**2)
        q = (1-p)/3
        both_flips  = (2*q<=rand) & (rand<3*q)
        if Z:
            x_flips =                rand<  q
            flips[:,:2*L**2] ^= x_flips
            flips[:,:2*L**2] ^= both_flips
        if X:
            z_flips =   (q<=rand) & (rand<2*q)
            off = 2*L**2 if Z else 0
            flips[:,off:off+2*L**2] ^= z_flips
            flips[:,off:off+2*L**2] ^= both_flips
        stabs = flips.dot(Ht)
        yield (stabs, flips)
