import numpy as np

class BayesianLogisticRegression:
    """
    Clase del modelo de Regresión Logística Bayesiana utilizando el algoritmo de Metropolis-Hastings (MCMC) para clasificación binaria.

    Este modelo permite:
    - Estimar la probabilidad de muerte de un nuevo paciente.
    - Evaluar la influencia de cada característica en el riesgo.
    - Estimar la incertidumbre en los parámetros aprendidos.
    """

    def __init__(self, num_samples=5000, step_size=0.01, burn_in=0.1, seed=42, init="random"):
        """
        Inicializa el modelo bayesiano.

        Parámetros:
        ----------
        num_samples : Número total de muestras MCMC a generar.
        step_size : Desviación estándar de la propuesta gaussiana (controla cuánto exploramos).
        burn_in : proporción de muestras iniciales a descartar (fase de calentamiento para olvidar el estado inicial).
        seed : Semilla para reproducibilidad.
        init : Estrategia de inicialización ('zero' o 'random').
        """
        self.num_samples = num_samples
        self.step_size = step_size
        if isinstance(burn_in, float) and 0 < burn_in < 1:
            self.burn_in = int(num_samples * burn_in)
        else:
            self.burn_in = int(burn_in)
        self.seed = seed
        self.init = init
        self.samples_ = None
        self.mean_weights_ = None

    def _sigmoid(self, z):
        """
        Función sigmoide para transformar la salida lineal en una probabilidad (0, 1).
        """
        return 1 / (1 + np.exp(-z))

    def _log_likelihood(self, X, y, weights):
        """
        Calcula la verosimilitud de los datos dados los pesos actuales.
        """
        z = np.dot(X, weights)
        p = self._sigmoid(z)
        return np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos usando MCMC con Metropolis-Hastings.

        El algoritmo funciona de la siguiente manera:
        1. Se inicia con pesos aleatorios.
        2. En cada iteración, se propone un nuevo conjunto de pesos perturbando los actuales con ruido gaussiano.
        3. Se calcula la probabilidad de aceptación comparando la verosimilitud de la propuesta vs la actual.
        4. Si se acepta, actualizamos los pesos; si no, mantenemos los anteriores.
        5. Se guardan las muestras después del periodo de 'burn-in'.

        Parámetros:
        ----------
        X : Matriz de características (n_samples x n_features).
        y : Vector de etiquetas binarias (0 o 1).
        """
        np.random.seed(self.seed)
        n_features = X.shape[1]
        
        # Inicialización de pesos
        if self.init == "zero":
            current_weights = np.zeros(n_features)
        else:
            current_weights = np.random.normal(loc=0.0, scale=1.0, size=n_features)
            
        current_ll = self._log_likelihood(X, y, current_weights)

        samples = []
        for _ in range(self.num_samples):
            proposal = current_weights + np.random.normal(0, self.step_size, size=n_features)
            proposal_ll = self._log_likelihood(X, y, proposal)

            log_ratio = proposal_ll - current_ll
            accept_prob = min(1, np.exp(log_ratio))
            
            # Decisión de aceptación
            if np.random.rand() < accept_prob:
                current_weights = proposal
                current_ll = proposal_ll

            samples.append(current_weights.copy())

        # Descartamos las primeras muestras (burn-in) para asegurar convergencia
        self.samples_ = np.array(samples[self.burn_in:])
        
        # Calculamos la media de los pesos posteriores para hacer predicciones puntuales
        self.mean_weights_ = self.samples_.mean(axis=0)

    def predict_proba(self, X):
        """
        Estima la probabilidad de clase positiva para nuevas muestras usando la media de los pesos posteriores.

        Parámetros:
        ----------
        X : Nuevas muestras (n_samples x n_features).

        Retorna:
        -------
        np.ndarray : Probabilidades estimadas.
        """
        if self.mean_weights_ is None:
            raise RuntimeError("El modelo debe ser ajustado antes de predecir.")
        return self._sigmoid(np.dot(X, self.mean_weights_))

    def predict(self, X, threshold=0.5):
        """
        Predice etiquetas binarias para nuevas muestras.

        Parámetros:
        ----------
        X : Nuevas muestras.
        threshold : Umbral de decisión para clasificar como 1.

        Retorna:
        -------
        np.ndarray : Etiquetas predichas (0 o 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def summary(self):
        """
        Devuelve la media y desviación estándar de los coeficientes (pesos).
        
        La desviación estándar nos da una medida de la incertidumbre de cada parámetro.

        Retorna:
        -------
        dict: Diccionario con media y desviación estándar por parámetro.
        """
        if self.samples_ is None:
            raise RuntimeError("El modelo aún no ha sido ajustado.")
        return {
            "mean": self.samples_.mean(axis=0),
            "std": self.samples_.std(axis=0)
        }
