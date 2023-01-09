# K-Nearest-Neighbors

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# When running directly
def train(neighbors):
    print("Loading Iris Dataset.... ", end="")
    data = datasets.load_iris()
    print("Done!")

    train_data, test_data, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.3)

    print(f"Training KNN with {neighbors} neighbors")
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_data, train_targets)

    print(f"Running predictions.... ", end="")
    pred = knn.predict(test_data)
    acc = metrics.accuracy_score(test_targets, pred)
    print(f"Accuracy: {acc}")


# For use with FastAPI
def predict(target, neighbors=5):
    data = datasets.load_iris()
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(data.data, data.target)
    pred = knn.predict([target])
    return data.target_names[pred[0]]


if __name__ == "__main__":
    train(5)