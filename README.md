# Machine Learning-Enhanced Temperature Sensing of Temperature-Insensitive Fluorophores

# Supporting Information

This repository contains the implementation code for the research paper "Machine Learning-Enhanced Temperature Sensing of Temperature-Insensitive Fluorophores" by Myeong Jin Kim and Jong Woo Lee.

This study demonstrates how machine learning algorithms can enhance temperature sensing using fluorescence spectroscopy, particularly with temperature-insensitive fluorophores like resveratrone. By extracting spectral features and applying various ML models, we achieved significant improvements in temperature prediction accuracy.

## Code Files
- **MLR**: Multiple Linear Regression model
- **SVR**: Support Vector Regression with RBF/polynomial kernels
- **LRF**: Linear Random Forest with linear models at leaf nodes
- **MLP+PDP_ICE**: Multilayer Perceptron with pyramid architecture. PDP and ICE analytics also run together.

## Dataset
Temperature range: 25-70°C

Features: 5 spectral features extracted from PL spectra

Performance metric: Mean Absolute Error (MAE) in °C

## Requirements
numpy, pandas, scikit-learn, tensorflow, matplotlib

```bibtex
@article{kim2024ml_temperature,
    title={Machine Learning-Enhanced Temperature Sensing of Temperature-Insensitive Fluorophores},
    author={Kim, Myeong Jin and Lee, Jong Woo},
    year={2025},
}
```

## Contact
Jong Woo Lee - promise@uos.ac.kr
