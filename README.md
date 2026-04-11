# LLMs Under the Microscope

This project re-examines public LLM rankings as statistical estimates rather than exact measurements. Using 4,496 models from Open LLM Leaderboard v2 and 33k battles from LMSYS Chatbot Arena, it measures how much of the published ordering actually survives uncertainty, finite sample size, and multiple comparisons.

## Project scope

The analysis treats benchmark scores as binomial proportions estimated over a fixed number of questions. From that starting point, the project applies confidence intervals, non-parametric bootstrap, pairwise significance tests with false-discovery-rate correction, power analysis, Bradley-Terry modeling, and Elo-style comparisons. It also studies how much of leaderboard performance can be explained by model metadata alone.

## Main findings

- GPQA, with only 448 questions, has a theoretical margin of error above +/-4.6 percentage points near 50%.
- More than 60% of top-100 pairs in GPQA are statistically indistinguishable.
- Multiple-comparison correction materially reduces the fraction of pairwise differences that remain significant at the top of the leaderboard.
- Metadata alone explains about R^2 = 0.70 of leaderboard average, suggesting that much of the visible ordering follows scale, architecture, precision, and producer effects.
- Arena and leaderboard agree qualitatively on the overlapping models, which supports the idea that large performance gaps are real while many fine-grained rank differences are mostly reporting noise.

## Core argument

The central conclusion is that exact leaderboard position is often the wrong unit of decision. For practical model selection, statistical tiers are more defensible than point-by-point rank comparisons.

## What this repository represents

The repository captures the project as a full empirical study: data preparation, statistical inference, ranking analysis, Arena modeling, final figures and tables, and the paper artifacts in English and Portuguese. Its main contribution is not another leaderboard, but a critique of how leaderboards are interpreted.
