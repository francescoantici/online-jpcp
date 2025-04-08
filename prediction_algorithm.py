class PredictionAlgorithm:

    def __init__(self, input_feature_set, target_feature, encoding_foo, classification_model, name = ""):
        self.name = name
        self.encoding_foo = encoding_foo
        self.classification_model = classification_model
        self.input_feature_set = input_feature_set
        self.target_feature = target_feature 
    
    def _data_preparation(self, jobs_data, is_train = None):
        """
        Prepares the data for prediction by selecting the relevant features.
        """
        # Select the relevant features from the jobs_data
        if is_train is not None:
            # Select the target feature if is in train mode
            return jobs_data[self.input_feature_set].values, jobs_data[self.target_feature].values
        else:
            # If in inference mode
            return jobs_data[self.input_feature_set].values
    
    def _feature_encoding(self, X):
        """
        Encodes the features using the specified encoding function.
        """
        # Encode the features using the encoding function
        return self.encoding_foo(X)
    
    def fit(self, jobs_data):
        """
        Fits the model to the training data.
        """
        # Prepare the data
        X, y = self._data_preparation(jobs_data = jobs_data, is_train=True)
        
        # Encode the features
        X = self._feature_encoding(X)
        
        # Fit the model
        self.classification_model.fit(X, y)
    
    def predict(self, jobs_data):
        """
        Predicts the target feature for the given test data.
        """
        # Prepare the data
        X = self._data_preparation(jobs_data = jobs_data, is_train=False)
        
        # Encode the features
        X = self._feature_encoding(X)
        
        # Predict using the model
        return self.classification_model.predict(X)