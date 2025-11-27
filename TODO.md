- Add rows missing pressure back in as soon as we stop using pressure as a parameter
- Things I learned:
  - Data cleaning made model more accurate (removing crazy pressure values)
  - Have to add ones after scaling or else determinant is zero which doesn't work
  - For sloppy variables, turns out periodicity is captured well by other variables, sin and cos of doy were nearly dependent

- Need to figure out how to actually evaluate models beyond normed errors