import numpy as np
EPS = 1e-12

class ComplexNMFbase:
    def __init__(self, n_basis=2, regularizer=0.1, eps=EPS):
        """
        Args:
            n_basis: number of basis
        """

        self.n_basis = n_basis
        self.regularizer = regularizer
        self.loss = []

        self.eps = eps

    def __call__(self, target, iteration=100, **kwargs):
        self.target = target

        self._reset(**kwargs)

        self.update(iteration=iteration)

        T, V = self.basis, self.activation
        Phi = self.phase

        return T.copy(), V.copy(), Phi.copy()

    def _reset(self, **kwargs):
        assert self.target is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_basis = self.n_basis
        n_bins, n_frames = self.target.shape

        self.basis = np.random.rand(n_bins, n_basis)
        self.activation = np.random.rand(n_basis, n_frames)
        self.phase = 2 * np.pi * np.random.rand(n_bins, n_basis, n_frames)

    def init_phase(self):
        n_basis = self.n_basis
        target = self.target

        phase = np.angle(target)
        self.phase = np.tile(phase[:, np.newaxis, :], reps=(1, n_basis, 1))

    def update(self, iteration=100):
        target = self.target

        for idx in range(iteration):
            self.update_once()

            TVPhi = np.sum(self.basis[:, :, np.newaxis] * self.activation[:, np.newaxis, :] * self.phase, axis=1)
            loss = self.criterion(TVPhi, target)
            self.loss.append(loss.sum())

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method")


class ComplexEUCNMF(ComplexNMFbase):
    def __init__(self, n_basis=2, regularizer=0.1, p=1, eps=EPS):
        """
        Args:
            n_basis: number of basis
        """
        super().__init__(n_basis=n_basis, eps=eps)

        self.criterion = lambda input, target: np.abs(input - target) ** 2
        self.regularizer, self.p = regularizer, p

    def _reset(self, **kwargs):
        super()._reset(**kwargs)

        self.init_phase()
        self.update_beta()

    def update(self, iteration=100):
        target = self.target

        for idx in range(iteration):
            self.update_once()

            TVPhi = np.sum(self.basis[:, :, np.newaxis] * self.activation[np.newaxis, :, :] * self.phase, axis=1)
            loss = self.criterion(TVPhi, target)
            self.loss.append(loss.sum())
            print(idx)

    def update_once(self):
        target = self.target
        regularizer, p = self.regularizer, self.p
        eps = self.eps

        T, V, Phi = self.basis, self.activation, self.phase
        Ephi = np.exp(1j * Phi)
        Beta = self.Beta
        Beta[Beta < eps] = eps

        X = T[:, :, np.newaxis] * V[np.newaxis, :, :] * Ephi
        ZX = target - X.sum(axis=1)

        Z_bar = X + Beta * ZX[:, np.newaxis, :]
        V_bar = V
        V_bar[V_bar < eps] = eps
        Re = np.real(Z_bar.conj() * Ephi)

        # Update basis
        VV = V ** 2
        numerator = (V[np.newaxis, :, :] / Beta) * Re
        numerator = numerator.sum(axis=2)
        denominator = np.sum(VV[np.newaxis, :, :] / Beta, axis=2)  # (n_bins, n_basis)
        denominator[denominator < eps] = eps
        T = numerator / denominator

        # Update activations
        TT = T ** 2
        numerator = (T[:, :, np.newaxis] / Beta) * Re
        numerator = numerator.sum(axis=0)
        denominator = np.sum(TT[:, :, np.newaxis] / Beta, axis=0) + regularizer * p * V_bar ** (
                    p - 2)  # (n_basis, n_frames)
        denominator[denominator < eps] = eps
        V = numerator / denominator

        # Update phase
        phase = np.angle(Z_bar)

        # Normalize basis
        T = T / T.sum(axis=0)

        self.basis, self.activation, self.phase = T, V, phase

        # Update beta
        self.update_beta()

    def update_beta(self):
        T, V = self.basis[:, :, np.newaxis], self.activation[np.newaxis, :, :]
        eps = self.eps

        TV = T * V  # (n_bins, n_basis, n_frames)
        TVsum = TV.sum(axis=1, keepdims=True)
        TVsum[TVsum < eps] = eps
        self.Beta = TV / TVsum