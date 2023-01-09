# Random Forest Classifier

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# When running directly
def train(estimators):
    print("Loading Iris Dataset.... ", end="")
    data = datasets.load_iris()
    print("Done!")

    train_data, test_data, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.3)

    print(f"Training Random Forest with {estimators} estimators")
    rfc = RandomForestClassifier(n_estimators=estimators)
    rfc.fit(train_data, train_targets)

    print(f"Running predictions.... ", end="")
    pred = rfc.predict(test_data)
    acc = metrics.accuracy_score(test_targets, pred)
    print(f"Accuracy: {acc}")


# For use with FastAPI
def predict(target, estimators=100):
    data = datasets.load_iris()
    rfc = RandomForestClassifier(n_estimators=estimators)
    rfc.fit(data.data, data.target)
    pred = rfc.predict([target])
    return data.target_names[pred[0]]


if __name__ == "__main__":
    for i in range(10):
        train((i+1)*10)
        print()