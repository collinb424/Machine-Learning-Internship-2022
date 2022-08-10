import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from scipy.optimize import curve_fit
from learningcurve import draw_samples, label_data, update_combined_dist

def parametric_fitting(num_data_available, mean_test_scores):
    # Fit the curve to the parametric model a*N^(-alpha) + b*N^(-beta) + c
    def parametric_model(N, a, b, c, alpha, beta):
        return a*N**(-1*alpha) + b*N**(-1*beta) + c

    param_bounds=([0,-np.inf,-np.inf,0,0],[np.inf,0,np.inf,1,1])
    popt, pcov = curve_fit(parametric_model, num_data_available, mean_test_scores, p0=[23.0, -22.35, 0.65, 0.43, 0.43], maxfev=500000, bounds=param_bounds)

    # Write results to a file
    opt = np.array2string(popt)
    fh = open('optimizedparams.txt', 'a')
    fh.write(opt)
    fh.write("\n")
    fh.close()

    aopt, bopt, copt, alphaopt, betaopt = popt
    N_model = np.logspace(1, 5, num=500, dtype=int)
    acc_model = parametric_model(N_model, aopt, bopt, copt, alphaopt, betaopt)
    return N_model, acc_model

def main():
    # Set up necessary variables
    total_test_scores = []
    total_true_test_scores = []
    n_splits = 10
    test_size = 0.3 # percentage of data used for testing
    num_data_available = np.logspace(1, 5, num=17, dtype=int)
    num_dimensions = 4
    num_extra_features = 20
    n_test = 1000

    # The outer for loop varies the total amount of data available to the algorithm
    for n in num_data_available:
        n_trials = 8
        trials_test_scores = []
        trials_true_test_scores = []

        # Use the specified number of trials to get a better average/estimate of the true accuracy
        for trial in range(n_trials):
            combined_dist = draw_samples(n, num_dimensions, num_extra_features)
            classification = label_data(n)
            external_combined_dist = draw_samples(n_test, num_dimensions, num_extra_features)
            external_classification = label_data(n_test)
            
            # Specify ML classifier and selector used for feature selection
            classifier = LogisticRegression(solver='lbfgs')
            selector = RFE(classifier, n_features_to_select=num_dimensions, step=1) # Uses recursive feature elimination

            # Perform feature selection and update the distributions
            # Note this is not done in the nested for loop since the same features would be selected since the data wouldn't have changed
            combined_dist = selector.fit_transform(combined_dist, classification)
            new_external_combined_dist = update_combined_dist(external_combined_dist, selector.ranking_)

            test_scores = []
            true_test_scores = []

            # Repeats the below process 10 times to mimic cross validation
            for i in range(n_splits):
                # Perform train_test_split after doing the above feature selection
                X_train, X_test, Y_train, Y_test = train_test_split(combined_dist, classification, test_size = test_size, shuffle=True)

                classifier.fit(X_train, Y_train)
                test_score = classifier.score(X_test, Y_test)
                true_test_score = classifier.score(new_external_combined_dist, external_classification)

                test_scores.append(test_score)
                true_test_scores.append(true_test_score)

            trials_test_scores.append(test_scores)
            trials_true_test_scores.append(true_test_scores)

        mean_trials_test_scores = np.mean(trials_test_scores, axis=1)
        mean_trials_true_test_scores = np.mean(trials_true_test_scores, axis=1)
        
        # Append the lists of train and test scores calculated above to a 2d array
        total_test_scores.append(mean_trials_test_scores)
        total_true_test_scores.append(mean_trials_true_test_scores)

    # Get mean scores
    mean_test_scores = np.mean(total_test_scores, axis=1)
    mean_true_test_scores = np.mean(total_true_test_scores, axis=1)

    N_model, acc_model = parametric_fitting(num_data_available, mean_test_scores)

    # Plot the Learning Curves with the curve fit model
    fig = plt.figure(1)
    plt.xscale("log")
    plt.plot(num_data_available, mean_true_test_scores, label='Non-Biased Test Accuracy', linewidth=2)
    plt.plot(num_data_available, mean_test_scores, label='Held Out Test Set Accuracy', linewidth=3.2)
    plt.plot(N_model, acc_model, 'k', alpha=0.9, label=r'Parametric Model ($aN^{-\alpha} + bN^{-\beta} + c$)', linewidth=2)
    plt.title('Learning Curve')
    plt.xlabel('Amount of Data Available')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('parametricfitting.png')

if __name__ == "__main__":
    main()