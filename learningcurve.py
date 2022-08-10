import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


def draw_samples(n, num_dimensions, num_extra_features):
    '''Generate n numbers from two Gaussian Distributions and then combine the results'''
    # Set parameters
    distribution_center = 0.2
    var = 0.8
    covar = 0.1

    pos_mean, neg_mean = generate_means(num_dimensions, num_extra_features, distribution_center)
    cov = create_covariance(num_dimensions, num_extra_features, var, covar)

    np.random.seed()
    pos_dist = np.random.multivariate_normal(pos_mean, cov, n//2).T
    neg_dist = np.random.multivariate_normal(neg_mean, cov, n//2).T

    combined_dist = create_combined_dist(pos_dist, neg_dist)
    return combined_dist


def generate_means(num_dimensions, num_extra_features, distribution_center):
    '''Generate d dimensional means, where d is the number of dimensions plus number of additional features'''
    pos_mean = [distribution_center]*num_dimensions
    neg_mean = [-1*distribution_center]*num_dimensions

    total_dimensions = num_dimensions + num_extra_features

    # For the additional features, add the same amount of noise as extra means for both distributions
    for i in range(num_dimensions, total_dimensions):    
        np.random.seed(3)
        noise = np.random.normal(0, 1)
        pos_mean.append(noise)
        neg_mean.append(noise)
    return pos_mean, neg_mean

def create_covariance(num_dimensions, num_extra_features, var, covar):
    '''Create a d dimensional covariance matrix, where d is the number of dimensions plus number of additional features'''
    total_dimensions = num_dimensions + num_extra_features
    cov_matrix = []
    for i in range(total_dimensions):
        cov_matrix.append([covar]*total_dimensions)
    
    # Set the diagonals equal to the specified variance
    for i in range(total_dimensions):
        cov_matrix[i][i] = var
    
    return cov_matrix

def create_combined_dist(pos_dist, neg_dist):
    '''Create a combined distribution of the positive and negative distributions'''
    combined_dist = []
    pos_dist = pos_dist.tolist()
    neg_dist = neg_dist.tolist()

    # Add the positive distribution to the combined distribution
    for i in range(len(pos_dist)):
        combined_dist.append(pos_dist[i])
    
    # Add the negative distribution to the combined distribution
    for i in range(len(neg_dist)):
        combined_dist[i] += neg_dist[i]

    combined_dist = np.array(combined_dist)
    return combined_dist.T

def label_data(n):
    '''Create the array with correct classifications (0 if in the first distribution, 1 if in the second distribution)'''
    if n % 2 == 0:
        classification = [0]*n
    else:
        classification = [0]*(n - 1)

    for i in range(n//2, len(classification)):
        classification[i] = 1

    classification = np.array(classification)
    return classification

def update_combined_dist(true_test_combined_dist, ranking):
    '''Updates true_test_combined_dist to have the same features as those in combined_dist'''
    new_true_test_combined_dist = []
    for i in range(len(ranking)):
        if ranking[i] == 1:
            new_true_test_combined_dist.append([row[i] for row in true_test_combined_dist]) # Appends the entire chosen column from reduced_combined_dist

    new_true_test_combined_dist = np.array(new_true_test_combined_dist)
    return new_true_test_combined_dist.T


def main():
    # Set up necessary variables
    total_test_scores = []
    total_true_test_scores = []
    n_splits = 10
    test_size = 0.3 # percentage of data used for testing
    num_data_available = np.logspace(1, 5, num=5, dtype=int)
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
            # Note this is not done in the nested for loop since the same features would be selected as the data wouldn't have changed
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

    # Plot the Learning Curves
    fig = plt.figure(1)
    plt.xscale("log")
    plt.plot(num_data_available, mean_test_scores, label='Held Out Test Set Accuracy')
    plt.plot(num_data_available, mean_true_test_scores, label='Non-Biased Test Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Amount of Data Available')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('learningcurve.png')

if __name__ == "__main__":
    main()