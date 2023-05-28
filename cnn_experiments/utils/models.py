import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform, xavier_normal


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, batch_norm=None, dropout=None):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions (you will need to add padding). Apply 2x2 Max
        # Pooling to reduce dimensions.
        # If P>N you should implement:
        # (Conv -> ReLU)*N
        # Hint: use loop for len(self.filters) and append the layers you need to the list named 'layers'.
        # Use :
        # if <layer index>%self.pool_every==0:
        #     ...
        # in order to append maxpooling layer in the right places.
        # ====== YOUR CODE: ======
        for i, out_channels in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
            
            if (i+1) % self.pool_every == 0:
                in_h //= 2
                in_w //= 2
                layers.append(nn.MaxPool2d(kernel_size=2))
        
        self.extracted_features = out_channels * in_h * in_w
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        in_features = self.extracted_features
        for out_features in self.hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input (using self.feature_extractor), flatten your result (using torch.flatten),
        # run the classifier on them (using self.classifier) and return class scores.
        # ====== YOUR CODE: ======
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        out = self.classifier(x)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims, batch_norm=None, dropout=None):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims, batch_norm, dropout)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()
        print(self)
    
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        for i, out_channels in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if self.batch_norm != None: 
                layers.append(nn.BatchNorm2d(out_channels))
            
                
            in_channels = out_channels
            
            if (i+1) % self.pool_every == 0:
                in_h //= 2
                in_w //= 2
                layers.append(nn.MaxPool2d(kernel_size=2))
                if self.dropout != None:
                    layers.append(nn.Dropout(self.dropout))
                
            
            
        
        self.extracted_features = out_channels * in_h * in_w
        # ========================
        seq = nn.Sequential(*layers)
        return seq
        
    def _make_classifier(self):
        in_channels, in_h, in_w = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # Hint: use loop for len(self.hidden_dims) and append the layers you need to list named layers.
        # ====== YOUR CODE: ======
        in_features = self.extracted_features
        for out_features in self.hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if self.batch_norm != None: 
                layers.append(nn.BatchNorm1d(out_features))
            if self.dropout != None:
                layers.append(nn.Dropout(self.dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================

