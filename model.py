from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# import mlflow
# from mlflow.models import infer_signature

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

class Model:
    def __init__(self, train, test, params=None):
        self.model = LogisticRegression(**params) if params else LogisticRegression(penalty='l2', C=0.1)
        # self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.target_variable = "Approve Loan"
        self.columns = train.drop(self.target_variable, axis=1).columns
        self.X_train = train.drop(self.target_variable, axis=1)
        self.y_train = train[self.target_variable].astype(int)
        self.X_test = test.drop(self.target_variable, axis=1)
        self.y_test = test[self.target_variable].astype(int)
        self.scaler = MinMaxScaler()

    def scaling_features(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        self.scaling_features()
        print(self.X_train)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)

        # Evaluate on test data
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        # Display the results
        print("Evaluation on test set")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        # Evaluate on train data (to check overfit)
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_train)

        # Evaluate on test data
        accuracy = accuracy_score(self.y_train, y_pred)
        conf_matrix = confusion_matrix(self.y_train, y_pred)
        class_report = classification_report(self.y_train, y_pred)

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

        X = np.concatenate((self.X_train, self.X_test), axis=0)
        y = np.concatenate((self.y_train, self.y_test), axis=0)

        # Perform cross-validation and print the results
        cross_val_results = cross_val_score(self.model, X, y, cv=kf,
                                            scoring='accuracy')  # Replace 'accuracy' with your chosen metric
        print(f'Cross-Validation Results: {cross_val_results}')
        print(f'Mean Accuracy: {cross_val_results.mean()}')

        # Get feature importance (absolute values of coefficients)
        feature_importance = abs(self.model.coef_[0])

        # Plot the feature importance
        plt.bar(self.columns, feature_importance)
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


