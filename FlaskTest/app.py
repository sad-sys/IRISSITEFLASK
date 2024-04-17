from flask import Flask, request, render_template
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25, random_state=0)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract the input features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = knn.predict(features)
        predicted_class = iris.target_names[prediction][0]

        return render_template('index.html', result=predicted_class)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
