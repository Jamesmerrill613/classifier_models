#Logistic Regression Classifier

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# When running directly
def train(rand_state):
    print("Loading Iris Dataset.... ", end="")
    data = datasets.load_iris()
    print("Done!")

    train_data, test_data, train_targets, test_targets = train_test_split(data.data, data.target, test_size=0.3)

    print(f"Training Logistic Regression with {rand_state} state")
    lrc = LogisticRegression(random_state=rand_state)
    lrc.fit(train_data, train_targets)

    print(f"Running predictions.... ", end="")
    pred = lrc.predict(test_data)
    acc = metrics.accuracy_score(test_targets, pred)
    print(f"Accuracy: {acc}")


# For use with FastAPI
def predict(target, rand_state=0):
    data = datasets.load_iris()
    lrc = LogisticRegression(random_state=rand_state)
    lrc.fit(data.data, data.target)
    pred = lrc.predict([target])
    return data.target_names[pred[0]]


if __name__ == "__main__":
    train(0)