## [Introduction]
This codebase is written for demonstration purposes to aid the research done by Dr. Spivey and his abstract titled "Use of AI Modeling Systems to Guide and Improve No Show Rates In a Pediatric Telemedicine / Virtual Sleep Medicine Clinic". Using synthetic data, this codebase trains 4 different model types and tests them at their accuracy in predicting clinical no-show rates using a number of factors (ranging from past-no shows to if the appointment is a weekday).

## [Pipeline]

Using a synthetic pateint data generator called synthea, I generated over 100,000 datapoints and filter for pediatrics only. Then, I set a target no-show rate at 20% and synthetically create no shows from random patients (setting START = NaT and show = 0). Then, I clean that data from the "raw" folder and place it in the "processed" folder (all of this is in /data). Then, I train 4 models (Logistic Regression, Elastic Net, Random Forest, and XGBoost) on the processed data set. The results are then placed in metrics.csv and graphics created are shown in the /metrics folder.

## [Recomendation]




## [Disclaimer]
This project is for demonstration only. The goal of the project is to give possible avenues that could be used on real data sets to predict no-show rates. All of thios data is synthetic and does not represent real trends.

In order to aid in this process, different AI tools where used to clean, debugg, and write code in different parts of this project.
