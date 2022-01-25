This page provides codes for l1 trend filtering based detection of short-term slow slip events.

Paper information:
- Title: l1 Trend Filtering-based Detection of Short-term Slow Slip Events: Application to a GNSS Array in Southwest Japan
- Authors: Keisuke Yano and Masayuki Kano
- Journal: in revision
- Plain language summary: Slow slip events characterized by a slower fault rupture compared to regular earthquakes have been discovered in tectonic zones worldwide and have helped us understand the surrounding stress environment including megathrust zone. This study focuses on short-term slow slip events with a duration of several days. These events have been observed by a Global Navigation Satellite System array, but do not often result in sufficient displacements that can be visually detected. So, refined detection methods have become necessary. We present a new automated detection method of short-term slow slip events using a Global Navigation Satellite System array. Our method utilizes l1 trend filtering, a sparse estimation technique, and combined p-value techniques to provide not only candidates of the events but also confidence values for detections. The synthetic tests showed that our method successfully detect almost all events with few misdetections. The application to real data in the Nankai subduction zone in western Shikoku, southwest Japan, revealed our method detected new potential events in addition to all known events. 


Usage (See also demo.ipynb):

- Import TrendFiltering_public as tf
- Import packages: numpy, random, pandas, scipy, statsmodels
- Use tf.trend_filtering(X_or,k=2,param_regularization,param_Lagrange)
- l1 trend filtering: l1 trend filtering is the fitting of piecewise linear time series without using the prior knowledge about knots (kink points)
-- Input: X, k, param_regularization, param_Lagrange
-- X: 1-dim sequence
-- k: fitting piecewise (k-1)-polynomial
-- param_regularization: a value of regularization hyperparameter (if you would like to select it, please check the 2nd output (Cp value) discussed below)
-- param_Lagrange: a value of optimization hyperparameter (no need to optimize; default value is 1)
-- Output: list consisting of (fitting result, Cp value, 2nd order difference of fitting result)
--- fitting result: fitting result
--- Cp value: criterion for selecting the regularization hyperparameter (smaller is preferable)
--- 2nd order difference of fitting result: check that this is a sparse vector
