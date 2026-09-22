# Dietary fatty acid balance shapes spatial learning and landmark use in honey bees

This repository contains the processed datasets and analysis code
accompanying the manuscript:

*Dietary fatty acid balance shapes spatial learning and landmark use in
honey bees*

Rozenbaum et al.

The repository is organized so that the principal analyses reported in
the manuscript can be reproduced using JMP, R, or Python. Processed
source data are in `data/`, data dictionaries are in `docs/`, and
analysis scripts are grouped by software in `JMP/`, `R/`, and `python/`.

## Repository structure

-   `data/` --- processed datasets used for the statistical analyses,
    together with selected simulation outputs.
-   `docs/` --- data dictionaries for the main processed datasets.
-   `JMP/` --- JMP scripts reproducing the maze, landmark-learning, and
    fatty-acid analyses performed in JMP.
-   `R/` --- R Markdown workflow reproducing the three binomial GLMM
    analyses of maze behavior.
-   `python/` --- Python scripts for the Trial-8 Monte Carlo analyses,
    the fitness-landscape sensitivity analysis, and the exact Trial-1
    random-search test for the landmark experiment.

## JMP analyses

The JMP scripts use relative paths and are intended to be run from the
`JMP/` folder of a downloaded or cloned copy of this repository. The
required `.jmp` data tables are stored in the sibling `data/` folder and
are opened automatically by the scripts; they do not need to be opened
manually first.

### Maze experiment

Run:

`JMP/Maze_JMP_analyses.jsl`

This script reproduces the JMP analyses of maze behavior, including
total arm entries, first mistake, magnitude and consistency of side
preference, finish time, distance flown, inter-trial intervals, the
Trial-1 treatment comparison, the relative-age covariate analysis, mean
right-turn proportion, and the maze markings-control experiment.

Required data tables:

-   `data/MazeLearning_Visits.jmp`
-   `data/MazeLearning_TimeAndDistance.jmp`
-   `data/MazeLearning_SidePreferenceConsistency.jmp`
-   `data/MazeLearning_InterTrialIntervals.jmp`

### Landmark-learning experiment

Run:

`JMP/Landmark_JMP_analyses.jsl`

This script reproduces the JMP analyses of landmark learning, including
inter-trial intervals, number of feeder visits, the relative-age
covariate analysis, time and distance to finish, overall performance
ranks, and feeder approach direction.

Required data tables:

-   `data/LandmarkLearning_Main.jmp`
-   `data/LandmarkLearning_InterTrialIntervals.jmp`

The exact Trial-1 comparison with random search is reproduced separately
in Python, as described below.

### Fatty-acid analyses

Run:

`JMP/FattyAcids_analyses.jsl`

This script reproduces the fatty-acid analyses reported in Fig. 6 and
Figs. S4--S5, including the seasonal analyses of the omega-6:3 ratio,
total essential fatty acids, omega-3, and omega-6.

Required data table:

-   `data/FattyAcids.jmp`

## R analyses

The three binomial GLMM analyses of maze behavior are reproduced in:

`R/Maze_binomial_GLMMs.Rmd`

The R Markdown file fits the models with the `glmmTMB` package and
reproduces analyses of:

1.  proportion of direction changes (Fig. 4c);
2.  proportion of transitions to an adjacent arm (Fig. 4d);
3.  proportion of arm entries in which bees turned back before reaching
    the feeder.

Required input files:

-   `data/PropSwitch.csv`
-   `data/Prop1stStep.csv`
-   `data/PropRegret.csv`

Required R packages:

-   `readr`
-   `glmmTMB`

To reproduce the analyses, open `R/Maze_binomial_GLMMs.Rmd` from within
the repository and Knit the document. The input paths are relative to
the `R/` folder.

## Python analyses

Python scripts and their principal input file are in `python/`.

### Trial-8 Monte Carlo analyses

Run from the repository:

``` bash
cd python
python MonteCarlo_trial8_CIs.py
```

The script reads:

`python/Transition_probabilities_trial8.csv`

and performs 100,000 pseudo-experiments for each treatment under the
empirical transition model, Random-8 model, and Random-7 low-reentry
model. It generates:

-   `python/MonteCarlo_summary.csv`
-   `python/MonteCarlo_distributions.csv`

For convenience, the repository also contains the corresponding
precomputed outputs in `data/`:

-   `data/MonteCarlo_summary.csv`
-   `data/MonteCarlo_distributions.csv.zip`

### Fitness-landscape sensitivity analysis

Run:

``` bash
cd python
python FitnessLandscape_trial8.py
```

The script reads `python/Transition_probabilities_trial8.csv` and
simulates performance across a grid of adjacent-arm transition
probabilities and direction-change probabilities. It generates the
numerical landscape and individual plot files in the current working
directory.

The precomputed numerical landscape used for the corrected supplementary
analysis is also provided as:

`data/fitness_landscape_actual_p1_pflip_piecewiseLinear.csv`

### Landmark Trial-1 exact random-search test

Run:

``` bash
cd python
python Landmark_trial1_random_search.py
```

The script reads `data/LandmarkLearning.csv` and reproduces the exact
one-sided Trial-1 comparison with random search. Under the null model,
each feeder visit has an independent probability of 1/8 of encountering
the rewarded feeder, with revisits allowed; the total number of visits
across bees is evaluated using the corresponding negative-binomial
distribution.

The script reports the number of bees, observed total visits, mean
visits, and exact one-sided P-value for each treatment.

Python packages used across these scripts include:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scipy`

## Processed datasets

The principal processed CSV datasets are:

-   `data/MazeLearning_visits.csv` --- trial-level measures derived from
    arm-visit sequences in the radial-maze experiment.
-   `data/MazeLearning_TimeAndDistance.csv` --- trial-level measurements
    of maze completion time and distance flown.
-   `data/LandmarkLearning.csv` --- trial-level data from the
    landmark-learning experiment.
-   `data/FattyAcids.csv` --- fatty-acid composition data.
-   `data/PropSwitch.csv`, `data/Prop1stStep.csv`, and
    `data/PropRegret.csv` --- inputs for the maze binomial GLMMs.

The `.jmp` files in `data/` are analysis-ready JMP tables used by the
reproducibility scripts. Some contain derived columns required for the
corresponding analyses.

## Data dictionaries

The `docs/` folder contains data dictionaries for the main processed
datasets:

-   `docs/MazeLearning_Visits_data_dictionary.xlsx`
-   `docs/MazeLearning_TimeAndDistance_data_dictionary.xlsx`
-   `docs/LandmarkLearning_data_dictionary.xlsx`
-   `docs/FattyAcids_data_dictionary.xlsx`

## Notes on reproducibility

The JMP, R, and Python workflows use the repository folder structure and
relative paths. The simplest approach is therefore to download or clone
the complete repository and retain the existing folder structure.

The repository contains processed analysis data rather than raw video
files. The processed datasets correspond to the variables used in the
statistical analyses reported in the manuscript.

Selected computationally intensive simulation outputs are supplied as
precomputed files so that the reported results can be inspected without
rerunning the full simulations.

## Citation

If you use the data or code in this repository, please cite the
associated publication:

Rozenbaum, E., Shrot, T., Daltrophe, H., Tietel, Z. and S. Shafir.
*Dietary fatty acid balance shapes spatial learning and landmark use in
honey bees.* Citation details will be updated upon publication.

If you use the datasets in this repository in your own work, please also
cite this GitHub repository using the repository URL.

## Contact

Questions about the repository or requests for additional information
may be directed to:

Sharoni Shafir\
B. Triwaks Bee Research Center\
Department of Entomology\
Institute of Environmental Sciences\
The Robert H. Smith Faculty of Agriculture, Food and Environment\
The Hebrew University of Jerusalem\
Rehovot, Israel

Email: sharoni.shafir@mail.huji.ac.il

## License

The data and code in this repository are made available under the
license provided in the accompanying `LICENSE` file.
