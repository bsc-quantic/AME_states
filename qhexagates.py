import numpy as np 
class QHexaGates:
    """
    A class to construct controlled operations between two quhexas (6-level systems).
    The controlled operation is applied only when the control quhexa is in a specific sublevel `j ∈ {0,...,5}`.
    The target operation is a diagonal phase gate: Ph = diag(e^{i a0}, e^{i a1}, ..., e^{i a5}),
    and is applied when the control qudit is in the state |j⟩.

    Example:
        If the control is in |1⟩, then:
        U = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Ph + |2⟩⟨2| ⊗ I + ... + |5⟩⟨5| ⊗ I
    """

    def __init__(self):
        self.q = [np.array([[int(i == j) for i in range(6)]]) for j in range(6)]
        self.I = np.eye(6, dtype=complex)

    def controlled_01(self, control_level: int, Ph: list):
        """
        Returns a controlled operation U acting on quhexas 0 and 2,
        where 0 is the control and 2 is the target.

        Parameters:
            control_level (int): Control sublevel j ∈ {0, 1, ..., 5}.
            Ph (list): Diagonal entries of the phase gate to apply on the target.

        Returns:
            np.ndarray: The resulting unitary matrix.
        """
        Ph = np.diag(Ph)
        U = 0j

        for i, q_i in enumerate(self.q):
            projector = np.dot(q_i.T, q_i)
            if i == control_level:
                U += np.kron(projector, np.kron(Ph, np.kron(self.I, self.I)))
            else:
                U += np.kron(projector, np.kron(self.I, np.kron(self.I, self.I)))
        return U

    def fourier(self):
        """Returns the 6-dimensional discrete Fourier transform matrix."""
        gate = np.ones((6, 6), dtype=complex)
        w = np.exp(-2j * np.pi / 6)
        for i in range(6):
            for j in range(6):
                gate[i, j] = w ** (i * j)
        return gate

    def P_gate_01(self):
        """
        Constructs a permutation-like gate acting on the first and third quhexas.
        Applies X^i on the target based on control level i.
        """
        X = np.eye(6, k=1)
        X[-1, 0] = 1
        P_gate = 0j
        for i, q_i in enumerate(self.q):
            X_i = np.linalg.matrix_power(X, i)
            P_gate += np.kron(np.dot(q_i.T, q_i), np.kron(X_i, np.kron(self.I, self.I)))
        return P_gate

    @staticmethod
    def X_0i(i: int):
        """
        Single-qudit X gate between sublevels 0 and i.

        Parameters:
            i (int): Sublevel to swap with 0. Must be in {1, 2, 3, 4, 5}.

        Returns:
            np.ndarray: The X_0i unitary matrix.
        """
        X = np.eye(6)
        X[0, 0] = X[i, i] = 0
        X[0, i] = X[i, 0] = 1
        return X

    def CNOT_initial_state(self, k: int, j: int):
        """
        Constructs a CNOT-like gate for preparing initial states.

        Parameters:
            k (int): Target qudit index (0 or 1).
            j (int): Control qudit index (2 or 3).

        Returns:
            np.ndarray: The resulting controlled gate matrix.
        """
        assert k in {0, 1} and j in {2, 3}, "k must be 0 or 1 and j must be 2 or 3"
        CNOT_is = 0j

        for i, q_i in enumerate(self.q):
            projector = np.dot(q_i.T, q_i)

            if i == 0:
                if k == 0:
                    CNOT_is += np.kron(projector, np.kron(self.I, np.kron(self.I, self.I)))
                else:  # k == 1
                    CNOT_is += np.kron(self.I, np.kron(projector, np.kron(self.I, self.I)))
            else:
                X_gate = QHexaGates.X_0i(i)
                if k == 0 and j == 2:
                    CNOT_is += np.kron(projector, np.kron(self.I, np.kron(X_gate, self.I)))
                elif k == 0 and j == 3:
                    CNOT_is += np.kron(projector, np.kron(self.I, np.kron(self.I, X_gate)))
                elif k == 1 and j == 2:
                    CNOT_is += np.kron(self.I, np.kron(projector, np.kron(X_gate, self.I)))
                elif k == 1 and j == 3:
                    CNOT_is += np.kron(self.I, np.kron(projector, np.kron(self.I, X_gate)))
        return CNOT_is

    @staticmethod
    def exp(angle: float):
        """Returns e^{2πi * angle / 6}"""
        return np.exp(2j * np.pi * angle / 6)

    @staticmethod
    def exp_3(angle: float):
        """Returns e^{2πi * angle / 3}"""
        return np.exp(2j * np.pi * angle / 3)
