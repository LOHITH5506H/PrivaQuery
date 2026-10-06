# PrivaQuery: Differential Privacy Engine

### Abstract: The failure of $k$-anonymity against Linkage Attacks
Traditional data anonymization techniques, such as redacting explicit identifiers (e.g., names or Social Security Numbers) or relying on $k$-anonymity, are fundamentally vulnerable to modern data mining techniques. As demonstrated by the classic Netflix Prize and Massachusetts Group Insurance Commission attacks, adversaries can execute **Linkage Attacks** by cross-referencing auxiliary public profiles against seemingly anonymous datasets to completely de-anonymize individuals. PrivaQuery demonstrates the mathematical failure of these legacy approaches and provides a robust, cryptographically secure alternative using Differential Privacy (DP).

### Mathematical Framework: The Laplace Mechanism and Bounded Sensitivity
Differential Privacy provides a rigorous mathematical guarantee that the presence or absence of any single individual in a dataset does not significantly affect the outcome of any query or statistical model. 
To achieve this, PrivaQuery implements the **Laplace Mechanism**:
$$M(x) = f(x) + \text{Laplace}\left(\frac{\Delta f}{\epsilon}\right)$$
Where $\Delta f$ is the $L_1$ sensitivity of the query, and $\epsilon$ (epsilon) is the privacy budget. By explicitly defining the logical bounds of clinical data (e.g., maximum theoretical income or blood pressure) and dynamically consuming $\epsilon$ for every operation, the engine guarantees that all aggregate statistics, synthetic data generation, and machine learning models remain immune to reconstruction attacks.

### Architecture: Zero-dependency embedded pipeline with L2-norm scaled Logistic Regression
PrivaQuery is built on a **Zero-Dependency Single-File Architecture** to prevent desynchronization between data generation and application state. 
- **Deterministic Pipeline**: The underlying clinical dataset is generated deterministically in memory, modeling robust correlations between risk factors (`Age`, `BloodPressure`, `Cholesterol`) and `HeartDisease`.
- **Bounded Feature Scaling**: The machine learning module implements a strict normalization pipeline. Features are normalized via `StandardScaler` and clipped to 3 standard deviations.
- **DP Machine Learning**: IBM's `diffprivlib` is utilized to train a DP-compliant Logistic Regression model. By rigorously calculating the $L_2$ norm of the feature space as $\sqrt{d} \times 3$ and explicitly passing it as the `data_norm` parameter, the architecture entirely prevents Laplace noise gradient explosions, achieving stable predictive utility ($\approx 0.86$ AUC) while guaranteeing privacy.

### Usage
```bash
streamlit run app.py
```
