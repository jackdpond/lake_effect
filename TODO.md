- Things I learned:
  - Data cleaning made model more accurate (removing crazy pressure values)
  - Have to add ones after scaling or else determinant is zero which doesn't work
  - For sloppy variables, turns out periodicity is captured well by other variables, sin and cos of doy were nearly dependent
  - For some reason, adding rows missing pressure back in as soon as we stop using pressure as a parameter made the model worse

  - The lake effect models had similar accuracy and f1 to the regular data, even with much more limited data.
  - In fact, the lake effect quantities were the steeper parameters
  - In lake effect, periodicity dropped out later

- Train the regular model on similar sized dataset and see how it performs
- Work on write up first, before anything extra.
- Optional extra models: Complicated markov, regression on amount of rain when it does rain, arima and sarima, even random forest
  - Arima and Sarima would be simple with statsmodels, I think
  - Use markov class for complicated markov
