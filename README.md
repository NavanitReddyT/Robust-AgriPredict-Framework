# Beyond Accuracy: A Climate-Aware Robustness Evaluation and Decision Support Framework for Crop Recommendation Systems

This repository contains the complete implementation codebase and research documentation for a climate-resilient agricultural decision support framework. While traditional crop recommendation models achieve near-perfect static accuracy on historical data, they lack insight into how predictions shift under environmental instability. This framework bridges that gap by stress-testing machine learning backbones against simulated climate changes.

---

## Abstract / Project Overview
Crop recommendation systems built on classical machine learning have reached near-perfect accuracy on standard benchmarks, yet they offer no insight into how their outputs hold up when environmental conditions shift. A model that confidently recommends rice under today's climate but has never been stress-tested against a $2^{\circ}C$ temperature rise or a 10% drop in rainfall is of limited value for real agricultural planning. 

Rather than proposing another static classifier, this framework wraps a standard high-accuracy machine learning backbone with four purpose-built modules to transition predictions into genuine decision support:
1. **Climate Perturbation Module:** Subjects the trained models to parameterized environmental stress scenarios (e.g., shifts in temperature, precipitation).
2. **Transition Matrix Analysis:** Tracks exactly which crops are displaced and by what substitutes under each scenario.
3. **Heuristic Counterfactual Engine:** Translates a desired crop outcome into minimal, actionable soil-amendment steps.
4. **Risk-Weighted Portfolio Recommender:** Replaces a single predicted label with a probability-ranked, diversified crop strategy.

---

## Dataset Specifications
The pipeline utilizes an agricultural benchmarking dataset containing **2,200 records** distributed equally across 22 distinct crop categories (100 samples per crop). 

### Features Analysed:
* **Soil Nutrients:** Nitrogen (N), Phosphorus (P), and Potassium (K) ratios.
* **Climatic Metrics:** Temperature (°C), Relative Humidity (%), and Rainfall (mm).
* **Soil Chemistry:** pH value.
* **Target Classes:** 22 crops, including Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, and Coffee.

---

## Core Methodologies & Pipeline
The underlying Jupyter Notebook (`MasterCode.ipynb`) executes a multi-stage data science and evaluation workflow:
* **Exploratory Data Analysis (EDA):** Profiling dataset class distributions, evaluating balance metrics, and calculating feature-to-class dependency distributions.
* **Correlation Modeling (Pre vs. Post Adversarial Attack):** Utilizing Pearson Correlation matrices to evaluate shifting linear relationships between environmental parameters before and after simulated environmental/adversarial data shocks.
* **Predictive Backbones:** Training robust classical classifiers to form the foundation of the recommendations.

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with Jupyter Notebook or JupyterLab.

### Installation
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/climate-aware-crop-recommendation.git](https://github.com/YOUR_USERNAME/climate-aware-crop-recommendation.git)
