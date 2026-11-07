from src.modeling.model_factory import ModelFactory


class ModelTrainer:

    @staticmethod
    def train(model_name, X_train, y_train):

        # Create the requested classifier
        model = ModelFactory.create(
            model_name,
            y_train=y_train
        )

        # Train the classifier
        model.fit(X_train, y_train)

        return model