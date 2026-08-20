import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List

class SimpleClassifier(nn.Module):
    """Simple neural network for classification tasks - compatible with Deep SHAP"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int] = [128, 64], 
        num_classes: int = 2, 
        dropout_rate: float = 0.2,
        activation: Optional[Union[str, nn.Module]] = 'relu'
    ):
        super(SimpleClassifier, self).__init__()
        
        # Map string to activation function if needed
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                activation_fn = nn.ReLU()
            elif activation.lower() == 'tanh':
                activation_fn = nn.Tanh()
            elif activation.lower() == 'sigmoid':
                activation_fn = nn.Sigmoid()
            elif activation.lower() == 'leakyrelu':
                activation_fn = nn.LeakyReLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif isinstance(activation, nn.Module):
            activation_fn = activation
        else:
            raise ValueError("activation must be a string or nn.Module")
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            to_add = [
                nn.Linear(prev_size, hidden_size),
                activation_fn,
            ]
            if dropout_rate > 0.0:
                to_add.append(nn.Dropout(dropout_rate))
            layers.extend(to_add)
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Store references to specific layers for SHAP analysis
        self.conv2 = None  # Placeholder for CNN layers if needed
        self.fc1 = layers[0]  # First linear layer
        self.fc2 = layers[-1]  # Output layer
        
    def forward(self, x):
        """Forward pass - returns single dimensional output as required by Deep SHAP"""
        if isinstance(x, list):
            # Handle list of tensors by concatenating
            x = torch.cat(x, dim=1)
        
        output = self.network(x)
        
        # Ensure single dimensional output for binary classification
        if output.shape[1] == 2:
            return output[:, 0:1]  # Return probability of positive class
        else:
            return output.squeeze()

class NeuralNetworkTrainer:
    """Neural Network Trainer compatible with Deep SHAP requirements - sklearn-style API"""
    
    def __init__(self, model_type: str = 'classifier', learning_rate: float = 0.001,
                 optimizer_type: str = 'adam', weight_decay = 0.0, epochs: int = 100, batch_size: int = 32,
                 verbose: bool = False, random_state: Optional[int] = None, **model_kwargs):
        """
        Initialize trainer with specified model type and training parameters

        Args:
            model_type: 'classifier'
            learning_rate: Learning rate for optimizer
            optimizer_type: 'adam' or 'sgd'
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
            random_state: seeds torch (weight init + DataLoader shuffling order) in fit(), for
                reproducible training. FIX: previously nothing seeded torch's RNG here, so two
                "identical" fit() calls (same data, same GSHAPA3 random_state - which only seeds
                the sklearn train_test_split, never reaches this class) produced different
                trained weights and therefore different SHAP outputs every run. Verified: two
                back-to-back fits on the same data gave visibly different gene-set-level
                significance in a compositional-normalization diagnostic (t=3.39 vs t=2.18 on
                the same 12 data points). Leave None to keep the old (unseeded) behavior.
            **model_kwargs: Arguments for model initialization
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        # FIX: weight_decay was accepted here but never stored or passed to the optimizer below
        # (_compile() built optim.Adam/SGD with no weight_decay arg at all) - it silently had zero
        # effect regardless of what was passed in.
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted_ = False
        
    def _initialize_model(self, input_size: int):
        """Initialize model with correct input size"""
        model_kwargs = self.model_kwargs.copy()
        if 'input_size' not in model_kwargs:
            model_kwargs['input_size'] = input_size
            
        # Initialize model based on type
        if self.model_type == 'classifier':
            self.model = SimpleClassifier(**model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model.to(self.device)
        
    def _compile(self):
        """Configure optimizer and loss function"""
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
            
        self.criterion = nn.BCEWithLogitsLoss()  # For binary classification
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the neural network to training data (sklearn-style API)
        
        Args:
            X: Training features (numpy array, tensor, or pandas DataFrame)
            y: Training labels (numpy array, tensor, or pandas Series)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            self: Returns the fitted estimator
        """
        # FIX: seed torch before weight init and DataLoader creation - see random_state note in
        # __init__. Must happen before _initialize_model() (weight init draws from torch's RNG)
        # and before the DataLoader is built (shuffle=True uses torch's RNG for sampling order).
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Convert DataFrames/Series to numpy if needed
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if X_val is not None and isinstance(X_val, (pd.DataFrame, pd.Series)):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, (pd.DataFrame, pd.Series)):
            y_val = y_val.values
        # Convert tensors to numpy if needed
        if torch.is_tensor(X):
            X = X.numpy()
        if torch.is_tensor(y):
            y = y.numpy()
        if X_val is not None and torch.is_tensor(X_val):
            X_val = X_val.numpy()
        if y_val is not None and torch.is_tensor(y_val):
            y_val = y_val.numpy()
        
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("This trainer is designed for binary classification only. "
                             "For multi-class, use a different model or modify the loss function."
                             )
       
        
        # Initialize model with correct input size
        if self.model is None:
            input_size = X.shape[1] if len(X.shape) == 2 else X.shape[1:]
            if isinstance(input_size, tuple):
                # For CNN, input_size might be (channels, height, width)
                if len(input_size) == 3:
                    self._initialize_model(input_size[0])  # channels
                else:
                    self._initialize_model(input_size)
            else:
                self._initialize_model(input_size)
        
        # Compile model
        self._compile()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X)
        y_train_tensor = torch.FloatTensor(y)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze(1)
                batch_y = batch_y 
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Validation
            val_acc = None
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
            
            if self.verbose and (epoch ) % max(1, self.epochs // 10) == 0:
                train_acc = 100 * correct / total
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%', end='')
                if val_acc is not None:
                    print(f', Val Acc: {val_acc:.2f}%')
                else:
                    print()
        
        self.is_fitted_ = True
        return self
    
    def _evaluate(self, data_loader):
        """Evaluate model on given data loader"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        self.model.train()
        return 100 * correct / total
    
    def get_model_for_shap(self, layer_name: Optional[str] = None) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """
        Get model in format required by Deep SHAP
        
        Args:
            layer_name: Name of layer to analyze (e.g., 'conv2', 'fc1')
            
        Returns:
            nn.Module for full model analysis, or (model, layer) tuple for layer-specific analysis
        """
        self.model.eval()  # Set to evaluation mode for SHAP
        
        if layer_name is None:
            # Return just the model for full analysis
            return self.model
        else:
            # Return (model, layer) tuple for layer-specific analysis
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                return (self.model, layer)
            else:
                available_layers = [name for name, _ in self.model.named_modules() if name]
                raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
    
    def _predict_proba(self, X):
        """
        Make predictions on new data (sklearn-style API)
        
        Args:
            X: Input features (numpy array or tensor)
            
        Returns:
            numpy array of predictions
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.numpy()
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs)
            
        return probabilities.cpu().numpy()
    
    def predict(self, X):
        probs = self._predict_proba(X)
        return (probs > 0.5)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (sklearn-style API)
        
        Args:
            X: Input features (numpy array or tensor)
            
        Returns:
            numpy array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if torch.is_tensor(X):
            X = X.numpy()
            
        probs = self._predict_proba(X)
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels (sklearn-style API)
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            float: Mean accuracy
        """
        predictions = self.predict(X)
        predicted_classes = (predictions > 0.5).astype(int)
        if torch.is_tensor(y):
            y = y.numpy()
        return np.mean(predicted_classes.flatten() == y)
    
    def get_layer_names(self) -> List[str]:
        """Get all available layer names for SHAP analysis"""
        return [name for name, _ in self.model.named_modules() if name]
    
    def get_params(self, deep=True):
        """Return parameters for this estimator."""
        params = {
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose,
        }
        # Include model_kwargs as separate params for grid search
        for k, v in self.model_kwargs.items():
            params[k] = v
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.model_kwargs[key] = value
        return self

if __name__ == "__main__":
    import shap
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)  # 1000 samples, 20 features
    y_train = (X_train.sum(axis=1) > 0).astype(float)  # Simple classification rule

    # Create and train model
    model = NeuralNetworkTrainer(
        model_type='classifier',
        hidden_sizes=[64, 32],
        #num_classes=2
    )

    model.fit(X_train, y_train)
    # Get model for SHAP - full model analysis
    model_for_shap = model.get_model_for_shap()
    print(f"Model for SHAP: {type(model_for_shap)}")

    # Get model for SHAP - layer-specific analysis
    model_layer_tuple = model.get_model_for_shap(layer_name='fc1')
    print(f"Model-layer tuple for SHAP: {type(model_layer_tuple)}")
    print(f"Model: {type(model_layer_tuple[0])}, Layer: {type(model_layer_tuple[1])}")

    # Available layers
    print(f"Available layers: {model.get_layer_names()}")

    # Test prediction
    test_sample = np.random.randn(5, 20)
    predictions = model.predict(test_sample) 

    #try to run shap explainer on model
    X_tensor = torch.from_numpy(X_train[:100]).float()
    X_test = torch.from_numpy(X_train[100:]).float()
    model_for_shap(X_tensor).shape

    explainer = shap.DeepExplainer(model_for_shap, X_tensor)
    shap_vals = explainer.shap_values(X_test[:5])  # Get SHAP values for test samples