###########################################################
### Imports
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

###########################################################
### Investigating SPX5.L

def investigate_univariate_timeseries(df, series):
  ts = df[series]
  # log returns
  log_returns = np.log(ts / ts.shift(1)).dropna() * 100
  # Autocorrelation Function (ACF)
  plot_acf(log_returns, lags=20)
  plt.title("Autocorrelation Function")
  plt.show()

  # Partial Autocorrelation Function (PACF)
  plot_pacf(log_returns, lags=20, method='ywm')  # 'ywm' = Yule-Walker Modified
  plt.title("Partial Autocorrelation Function")
  plt.show()

###########################################################
### ARMA models

def investigate_ARMA_models(df, series):
  # split in/out-of-sample
  #df.index = pd.to_datetime(df.index)
  in_sample_df = df[df['Date'] < '2025-02-01 00:00:00']
  out_of_sample_df = df[df['Date'] >= '2025-02-01 00:00:00']
  print('in sample head:\n ' + str(in_sample_df.head()))
  print('out of sample head:\n' + str(out_of_sample_df.head()))

  in_sample_series = in_sample_df[series]
  out_of_sample_series = out_of_sample_df[series]

  #Store results
  results = []

  # Loop over p and q
  for p in range(2,3):
      for q in range(1,2):
          try:
              # Fit on training data
              print(f'training {p} {q}')
              model = ARIMA(in_sample_series, order=(p, 0, q))
              fit = model.fit(cov_type='robust')
              print(fit.summary())
              print(fit.test_serial_correlation(method='ljungbox', lags=3))

              # In-sample evaluation
              print('in sample pred')
              #in_sample_preds = fit.predict()
              #in_sample_mse = mean_squared_error(in_sample_series, in_sample_preds)
              llf = fit.llf  # Log-likelihood
              aic = fit.aic
              bic = fit.bic

              # Ljung-Box test on residuals (for autocorrelation)
              print('LB')
              lb_test = acorr_ljungbox(fit.resid, lags=[3], return_df=True)
              lb_stat = lb_test['lb_stat'].values[0]
              lb_pvalue = lb_test['lb_pvalue'].values[0]

              # Out-of-sample prediction
              print('out of sample pred')
              forecast = fit.forecast(steps=len(out_of_sample_series))
              out_sample_mse = mean_squared_error(out_of_sample_series, forecast)

              # Store results
              results.append({
                  'p': p,
                  'q': q,
                  'aic': aic,
                  'bic': bic,
                  'loglik': llf,
                  #'in_sample_mse': in_sample_mse,
                  'out_sample_mse': out_sample_mse,
                  'ljung_box_stat': lb_stat,
                  'ljung_box_pvalue': lb_pvalue
              })

          except Exception as e:
              results.append({
                  'p': p,
                  'q': q,
                  'aic': None,
                  'bic': None,
                  'loglik': None,
                  #'in_sample_mse': None,
                  'out_sample_mse': None,
                  'ljung_box_stat': None,
                  'ljung_box_pvalue': None,
                  'error': str(e)
              })

  # Convert to DataFrame
  results_df = pd.DataFrame(results)
  results_df = results_df.sort_values(by='aic', na_position='last')

  # Display best models
  pd.set_option("display.float_format", "{:.4f}".format)
  print(results_df)

###########################################################
### VAR modelling

# TODO: Add in/out of sample split
def investigate_VAR_models(df):
  var_df = df[['SPX5.L', 'SPY5l.AQX', 'SPY5.MIL']]

  # Fit VAR model
  model = VAR(var_df)
  results = model.fit()  # automatic lag selection using AIC

  # Summary
  print(results.summary())
  residuals = results.resid

  # Check residuals
  print(residuals.head())
  residual_corr = residuals.corr()
  print("Residual Correlation Matrix:\n", residual_corr)

  # Heatmap of residual correlation
  sns.heatmap(residuals.corr(), annot=True, cmap='coolwarm')
  plt.title("Residual Correlation Matrix")
  plt.show()

###########################################################
### main
def main():
  df = pd.read_csv('output/sp_g2.csv')

  series_to_investigate = 'SPX5.L'
  #investigate_univariate_timeseries(df, series_to_investigate)

  investigate_ARMA_models(df, series_to_investigate)

  #investigate_VAR_models(df)

  return


###########################################################
### start main
if __name__ == "__main__":
  main()