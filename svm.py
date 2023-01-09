# Support Vector Machine

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# When running directly
def train(kernel):
    print("Loading Iris Dataset.... ", end="")
    data = datasets.load_iris()
    print("Done!")

    train_data, test_data, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.3)

    print(f"Training Support Vector Machine with {kernel} kernel")
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_data, train_targets)

    print(f"Running predictions.... ", end="")
    pred = clf.predict(test_data)
    acc = metrics.accuracy_score(test_targets, pred)
    print(f"Accuracy: {acc}")


# For use with FastAPI
def predict(target, kernel='linear'):
    data = datasets.load_iris()
    clf = svm.SVC(kernel=kernel)
    clf.fit(data.data, data.target)
    pred = clf.predict([target])
    return data.target_names[pred[0]]


if __name__ == "__main__":
    train('linear')