import numpy as np
from scipy import linalg as la
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        transition_matrix(np.array()) : nxn matrix of state transition probabilities
        labels(list(str)): List of state labels
        label_dict(dict(str:int)): Dictionary, keys are state labels, values are corresponding column index

    """

    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Check to see that the columns sum to zero; if not, raise ValueError
        checker = np.array(np.sum(A, axis=0))
        if not np.all(np.isclose(checker, 1)):
            raise ValueError('Columns must sum to one')
        
        # Set the transition attribute to A
        self.transition_matrix = A

        # If there is no states passed in, set it to a list of consecutive integers whose length corresponds to the size of A
        if not states:
            states = list(range(A.shape[0]))

        # Set the labels and label dict attributes
        self.labels = states
        self.label_dict = {label: n for label, n in zip(states, range(len(states)))} # The value of each label-key is the index of the column corresponding to the label


    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Set col variable to the index of the column of the transition matrix corresponding to the given state
        col = self.label_dict[state]

        # Get the categorical distribution and choose randomly from the distribution
        cat_dist = np.array(self.transition_matrix[:, col])
        new_state_array = np.random.multinomial(1, cat_dist)

        # Set the new state to the label corresponding to the selected row
        new_state_index = np.argmax(new_state_array)
        new_state = self.labels[new_state_index]

        return new_state


    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        # Initialize the list of steps containing the starting state
        steps = [start]
        
        for i in range(N-1): # Call transition to get the next step to add to the list, N-1 times
            steps.append(self.transition(steps[-1]))

        return steps


    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Initialize the list of steps containing the starting state
        steps = [start]
        while stop not in steps: # Call transition to add the next step to the list of steps until adding the stop state
            steps.append(self.transition(steps[-1]))
        return steps

    def steady_state(self, tol=1e-12, maxiter=100):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # Make a random vector whose two entries add up to 1
        x0 = np.random.random(len(self.labels))
        x0 = x0 / np.sum(x0)

        iter = 0
        while iter <= maxiter: # While iterations is less than maxiter
            xt = self.transition_matrix.dot(x0)
            if la.norm((xt-x0), 1) < tol: # Check if the newly transformed vector is pretty much the same as the former vector
                return xt # If so, we have reached a steady state and can return
            x0 = xt # If not, set x0 and xt for the next iteration
            iter += 1
        
        raise ValueError('Did not converge')
    

class LogReg:

    def __init__(self, intercept=True, scale=True):
        self.intercept = intercept
        self.scale= scale

    def obj_(self, beta, X, y):
        beta = np.asarray(beta)
        z = X @ beta
        # negative log-likelihood, stable form
        # l = sum( log(1 + exp(z)) - y*z )
        return np.sum(np.logaddexp(0, z) - y * z)
    
    def grad_(self, beta, X, y):
        beta = np.asarray(beta)
        z = X @ beta
        p = 1 / (1 + np.exp(-z))  # sigmoid
        # gradient: X^T (p - y)
        return X.T @ (p - y)
    
    def hessian_(self, beta, X):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        return X.T * w @ X  # broadcasting trick; still Xᵀ W X
    
    def fit(self, X, y, beta0=None, verbose=True):
        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        if self.intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

        if beta0 is None:
            beta0 = np.zeros(X.shape[1])


        res = minimize(
            self.obj_,
            beta0,
            args=(X_train, y_train),
            jac=self.grad_,              # <--- supply gradient
            method="L-BFGS-B",
            options={"maxiter": 1000}
        )

        # Get results of minimization
        self.message = res.message
        self.success = res.success
        self.beta = res.x

        # Calculate normed error
        beta = res.x
        z = X_test @ beta
        self.pred = 1 / (1 + np.exp(-z))
        self.normed_error = la.norm(self.pred - y_test)

        # Calculate hessian at minimizer and eigenvalues
        H = self.hessian_(beta, X)
        self.hessian_eigenvals = np.real(la.eigvals(H))

        if verbose:
            print(self.message)
            print("success:", self.success)
            print("beta:", self.beta)
            print("-" * 100)
            print(f"Normed Error: {self.normed_error}")
            print("-"* 100)
            print("Eigenvalues: ", self.hessian_eigenvals)