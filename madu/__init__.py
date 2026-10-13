"""Madu workload predictor (SoCC'22: "The Power of Prediction").

Seq2Seq LSTM encoder-decoder with data-dependent uncertainty learning
(Gaussian NLL, Eq.1) and stochastic attention (Eq.3-6). Final prediction
Y_hat = mu + sigma (Eq.8). Training/evaluation only (no deployment).
"""
