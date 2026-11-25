# IPL Winner Prediction

A beginner-friendly end to end machine learning baseline project in Python that predicts IPL match winners using pre-match context features.
## Dataset
- Custom dataset (`matches.csv`) containing 40 IPL 2026 matches with 9 columns (`match_id`, `season`, `city`, `team1`, `team2`, `toss_winner`, `toss_decision`, `venue`, `winner`).

## Project Files
- `matches.csv` — Match dataset used for exploration and training.
- `explore.py` — Checks dataset shape, columns, missing values, winner counts, and separates features (`X`) from target (`y`).
- `train.py` — Encodes text categories with `LabelEncoder`, splits data with `train_test_split`, trains a `RandomForestClassifier`, checks accuracy, and predicts a sample match winner.

## Notes & Limitations
- Built as a small baseline prototype on 40 matches across 10 teams, so test accuracy is low and can be improved with more historical match data and extra features.
