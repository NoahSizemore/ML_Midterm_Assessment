| experiment | n_features | learning_rate | iterations | val MSE | val RMSE |
|---|---|---|---|---|---|
| A: Previous Scores only | 1 | 0.1000 | 161 | 65.5626 | 8.0971 |
| B: A + Hours Studied | 2 | 0.1000 | 161 | 11.3575 | 3.3701 |
| C: 5 core features | 5 | 0.1000 | 161 | 8.3758 | 2.8941 |
| D: C + Sleep Hours Sq | 6 | 0.1000 | 162 | 4.2786 | 2.0685 |
| E: D + Weekly Study Hours | 7 | 0.1000 | 1155 | 4.2786 | 2.0685 |
| F: D + Commute Minutes | 7 | 0.1000 | 162 | 4.2906 | 2.0714 |
| G: all 7 + Sleep Hours Sq | 8 | 0.1000 | 1155 | 4.2906 | 2.0714 |
| D: C + Sleep Hours Sq (lr sweep) | 6 | 0.0010 | 14459 | 4.2786 | 2.0685 |
| D: C + Sleep Hours Sq (lr sweep) | 6 | 0.0100 | 1557 | 4.2786 | 2.0685 |
| D: C + Sleep Hours Sq (lr sweep) | 6 | 0.1000 | 162 | 4.2786 | 2.0685 |
| D: C + Sleep Hours Sq (lr sweep) | 6 | 0.5000 | 28 | 4.2786 | 2.0685 |
