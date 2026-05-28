# What Makes a Spotify Track Popular?

**Name(s):** Saaz Mahadakar  
**Course:** DSC 80 Final Project

## Introduction

This project uses Spotify track-level and artist-level metadata to study what is associated with track popularity. I focus on five musically distinct genres: `classical`, `country`, `hip-hop`, `jazz`, and `metal`.

My central question is:

**Among these five genres, which audio and metadata features are associated with higher Spotify popularity?**

The cleaned analysis dataset has **5,000 rows** (1,000 tracks per selected genre). Key columns for this question include:

- `popularity`: Spotify popularity score (0-100).
- `track_genre`: genre label.
- `danceability`, `energy`, `valence`, `acousticness`, `loudness`, `tempo`: core audio descriptors.
- `explicit`: whether Spotify labels the track as explicit.
- `release_year`, `duration_min`: metadata derived from raw columns.
- `artist_followers`, `artist_popularity`: artist-level context from the artist table.

## Data Cleaning and Exploratory Data Analysis

I cleaned the data with the following steps:

1. Loaded `music_tracks.csv` and `artists.csv`.
2. Removed the export artifact column `Unnamed: 0`.
3. Filtered to 5 selected genres (`classical`, `country`, `hip-hop`, `jazz`, `metal`).
4. Parsed `release_date` into `release_year`; created `duration_min` from milliseconds.
5. Built a standardized `primary_artist` key from the first listed artist per track.
6. Joined artist-level followers/popularity using artist name keys.
7. Kept all rows after merge, allowing unmatched artists to remain as missing values.

Head of cleaned DataFrame (subset of columns):

| track_name             | track_genre   |   popularity |   explicit |   energy |   danceability |   tempo |   release_year |
|:-----------------------|:--------------|-------------:|-----------:|---------:|---------------:|--------:|---------------:|
| Zara Zara              | classical     |           58 |          0 |    0.268 |          0.643 | nan     |           2001 |
| Kajra Re               | classical     |           59 |          0 |    0.898 |          0.484 | nan     |           2005 |
| Zara Zara - Lofi       | classical     |           54 |          0 |    0.638 |          0.608 | 140.109 |           1984 |
| Vaseegara              | classical     |           68 |          0 |    0.293 |          0.695 | nan     |           1972 |
| Zara Zara - LoFi Chill | classical     |           59 |          0 |    0.308 |          0.583 | 118.226 |           1987 |

Univariate visualization (required):

<iframe
  src="assets/popularity_distribution.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>

Popularity is strongly right-skewed in this subset: most tracks are low-to-mid popularity, with relatively fewer highly popular tracks.

Bivariate visualization (required):

<iframe
  src="assets/danceability_vs_popularity.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>

Danceability tends to have a mild positive association with popularity, but the relationship differs by genre, suggesting genre confounding.

Additional bivariate visualization aligned to hypothesis:

<iframe
  src="assets/explicit_popularity_boxplot.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>

Explicit tracks are concentrated in higher-popularity ranges in this five-genre subset, motivating the formal test in Step 4.

Interesting aggregate table (required):

| track_genre   |   mean_popularity |   mean_energy |   explicit_rate |   missing_tempo_rate |
|:--------------|------------------:|--------------:|----------------:|---------------------:|
| metal         |            43.705 |         0.84  |           0.142 |                0.097 |
| hip-hop       |            37.759 |         0.683 |           0.319 |                0.176 |
| country       |            17.028 |         0.597 |           0.03  |                0.218 |
| jazz          |            13.628 |         0.353 |           0.003 |                0.312 |
| classical     |            13.055 |         0.19  |           0     |                0.349 |

The aggregate table shows strong genre differences in both popularity and audio energy, plus large variation in tempo missingness rates.

## Assessment of Missingness

For `tempo` missingness, I ran permutation tests against:

- `duration_min` (numeric): observed absolute mean difference = **0.124**, p-value = **0.066**.
- `track_genre` (categorical, TVD statistic): observed TVD = **0.226**, p-value = **0.000**.

Interpretation:

- There is no strong evidence that `tempo` missingness depends on duration.
- There is very strong evidence that `tempo` missingness depends on genre.
- This supports a **MAR-like mechanism** for tempo missingness with respect to observed genre.

Potential NMAR note: columns like `album_name` could be NMAR if missingness is driven by metadata ingestion quality not present in the dataset; additional Spotify pipeline metadata would help evaluate this.

Related missingness plot:

<iframe
  src="assets/missingness_duration_plot.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>

## Hypothesis Testing

I test whether explicit labeling is associated with popularity in this selected-genre dataset.

- **Null hypothesis:** Explicit and non-explicit tracks have the same average popularity.
- **Alternative hypothesis:** Explicit tracks have higher average popularity.
- **Test statistic:** Difference in sample means, `mean(popularity | explicit) - mean(popularity | non-explicit)`.
- **Significance level:** 0.05.

Results from a permutation test:

- Observed mean difference = **8.261** popularity points.
- p-value = **0.000** (rounded).

Conclusion: I reject the null hypothesis. In this selected subset, explicit tracks are associated with higher average popularity.

<iframe
  src="assets/hypothesis_null_distribution.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>

## Framing a Prediction Problem

Prediction target:

- `is_popular = 1` if `popularity >= 70`, else `0`.

This is a **binary classification** problem.

Why this target:

- It matches the central popularity theme.
- It creates a decision-relevant outcome ("likely high popularity or not").

Metric choice:

- Primary metric: **F1 score** because high-popularity tracks are the minority class.
- I also report accuracy for context.

Prediction-time justification:

- Features are restricted to audio and metadata plausibly available when scoring a track (genre label, explicit label, duration, release year, and audio descriptors).
- I do not use any future-only outcomes as input.

## Baseline Model

Baseline pipeline:

- Numeric features: median imputation + standard scaling.
- Categorical features (`track_genre`): mode imputation + one-hot encoding.
- Binary feature (`explicit`): passthrough.
- Classifier: logistic regression with balanced class weights.

Feature types used:

- Quantitative: 12 (`duration_min`, `release_year`, and 10 audio columns).
- Nominal: 1 (`track_genre`).
- Binary: 1 (`explicit`).

Performance on held-out test data:

- Accuracy = **0.612**
- F1 = **0.287**

This baseline is reasonable but leaves room for improvement, especially on minority-class detection.

## Final Model

Improvements over baseline:

1. Added engineered interaction features (pairwise interactions among `danceability`, `energy`, `valence`).
2. Switched to a tuned Random Forest classifier.
3. Performed `GridSearchCV` for:
   - `n_estimators` in {200, 400}
   - `max_depth` in {8, 16, None}
   - `min_samples_leaf` in {1, 3, 8}

Best hyperparameters:

- `max_depth = 8`
- `min_samples_leaf = 3`
- `n_estimators = 200`

Final performance:

- Accuracy = **0.759**
- F1 = **0.321**

Improvement:

- F1 increased by **+0.034** over baseline.

## Fairness Analysis

I evaluated whether model performance differs by genre grouping:

- **Group X:** `hip-hop` and `metal`
- **Group Y:** `classical`, `country`, and `jazz`

Metric: **F1 score**.

- **Null hypothesis:** Model F1 is the same across groups up to chance.
- **Alternative hypothesis:** Model F1 is lower for Group X.
- **Test statistic:** `F1_Y - F1_X` (positive means worse for Group X).

Observed values:

- `F1_X = 0.345`
- `F1_Y = 0.203`
- Observed statistic = **-0.141**
- p-value = **0.985**

Conclusion: I fail to reject the null hypothesis; there is no statistical evidence that Group X is disadvantaged under this metric in this test split.

<iframe
  src="assets/fairness_null_distribution.html"
  width="900"
  height="550"
  frameborder="0"
></iframe>
