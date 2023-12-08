from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


class Model:
    def __init__(self, train, test, params=None):
        self.model = LogisticRegression(**params) if params else LogisticRegression(penalty='l2', C=0.1)
        self.target_variable = "Approve Loan"
        self.train_data = train
        self.test_data = test

    def train_model(self):
        X_train = self.train_data.drop(self.target_variable, axis=1)
        y_train = self.train_data[self.target_variable].astype(int)

        self.model.fit(X_train, y_train)

    def evaluate(self):
        X_test = self.test_data.drop(self.target_variable, axis=1)
        y_test = self.test_data[self.target_variable].astype(int)
        # Make predictions on the test data
        y_pred = self.model.predict(X_test)

        # Evaluate on test data
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Display the results
        print("Evaluation on test set")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        # Evaluate on train data (to check overfit)
        X_train = self.train_data.drop(self.target_variable, axis=1)
        y_train = self.train_data[self.target_variable].astype(int)
        # Make predictions on the test data
        y_pred = self.model.predict(X_train)

        # Evaluate on test data
        accuracy = accuracy_score(y_train, y_pred)
        conf_matrix = confusion_matrix(y_train, y_pred)
        class_report = classification_report(y_train, y_pred)

        # Display the results
        print("Evaluation on train set")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        print("Cross validation evaluation")

        num_folds = 20
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        grouped = pd.concat([self.train_data, self.test_data], ignore_index=True)
        X = grouped.drop(self.target_variable, axis=1)
        y = grouped[self.target_variable]

        # Perform cross-validation and print the results
        cross_val_results = cross_val_score(self.model, X, y, cv=kf,
                                            scoring='accuracy')  # Replace 'accuracy' with your chosen metric
        print(f'Cross-Validation Results: {cross_val_results}')
        print(f'Mean Accuracy: {cross_val_results.mean()}')

        # Get feature importance (absolute values of coefficients)
        feature_importance = abs(self.model.coef_[0])

        # Plot the feature importance
        plt.bar(X.columns, feature_importance)
        plt.title('Feature Importance in Logistic Regression')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Values')
        plt.xticks(rotation=60)
        plt.show()

        # self.experiment(accuracy, X_train)

    def inference(self, X):
        return self.model.predict(X)

    def experiment(self, accuracy, X_train):
        mlflow.set_experiment("MLflow Quickstart")

        with mlflow.start_run():
            # Log the hyperparameters
            # mlflow.log_params(params)

            # Log the loss metric
            mlflow.log_metric("accuracy", accuracy)

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")

            # Infer the model signature
            signature = infer_signature(X_train, self.model.predict(X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="loan_model",
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-quickstart",
            )


