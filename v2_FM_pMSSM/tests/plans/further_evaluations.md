# Further evaluations (not implemented yet)

Ideas beyond `tests/compare_fm.ipynb` (E1-E6). Each can become a new section there.

## Physics consistency of generated spectra
Check that generated points are physically possible spectra:
- neutralino mass ordering |m_chi_10| <= |m_chi_20| <= |m_chi_30| <= |m_chi_40|, and m_chi_1p <= m_chi_2p;
- chargino masses > 0 (SLHA convention; neutralino masses may be negative);
- the sign of a bino-like m_chi_10 equals sign(IN_M_1) (wino-like: sign(IN_M_2)).
Report the fraction of generated points violating each rule (the test data has 0 %).

## Posterior calibration (SBC / TARP)
For many test points: generate p(inputs | observables) and check that the true inputs fall into the
x % credible region x % of the time (SBC rank histograms per input, or TARP coverage for all inputs
together). This is what matters when the model is used for inference.

## Sharpness of p(observable | inputs)
Omega and the masses are deterministic functions of the inputs. Draw K samples per test point and compare
the spread of the K samples with the actual error: a well-calibrated model is narrow where it is right,
and its spread is about as large as its error.

## C2ST (classifier two-sample test)
Train a small classifier to tell generated points from test points, for the joint distribution and for
each mask. Accuracy 0.5 = indistinguishable. One number per run and mask, and it also sees correlations,
which the 1D marginals of E3 hide.

## Solver convergence
Sample the same jobs with step_size 0.05 and 0.02 (or another solver): if the results change, part of the
error is the ODE solver, not the model.

## Correlation-matrix difference
Correlation matrix of generated vs. test points, shown as the difference: a cheap overview of which
pairs of variables the model gets wrong, without a full corner plot.

## Prior box
Fraction of generated inputs outside the scan ranges (e.g. IN_M_3 outside [1000, 5000]): the model
should not leave the prior volume it was trained on.
