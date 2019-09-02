import pandas
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def data_information(dataset):
    #Dimensions of dataset
    print(dataset.shape)

    #Printing out the data
    print(dataset.head(20))

    #Descriptions
    print(dataset.describe())

    #Class destribution
    print(dataset.groupby('class').size())

def data_plotting(dataset):
    #Box and whisket plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    #Histograms
    dataset.hist()
    plt.show()

    #Multivariable plots
    scatter_matrix(dataset)
    plt.show()

def validation_dataset(dataset):
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

def main():
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv('iris.data', names=names)

    #data_information(dataset)
    #data_plotting(dataset)

    #Validation dataset
    x_train, x_validation, y_train, y_validation = validation_dataset(dataset)

    seed = 7
    scoring = 'accuracy'

    #Build Models
    #Spot check algoriths
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNC', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    #Evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: \t %f \t (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    print("\n")
    #Make predictions on validate dataset
    svm = SVC(gamma='auto')
    svm.fit(x_train,y_train)
    predictions = svm.predict(x_validation)

    print(accuracy_score(y_validation, predictions))
    print(confusion_matrix(y_validation, predictions))
    print(classification_report(y_validation, predictions))

    
if __name__ == "__main__":
    main()