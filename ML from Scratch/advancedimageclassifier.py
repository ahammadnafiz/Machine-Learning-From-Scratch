import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from ultralytics import YOLO

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, label

class AdvancedImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None, random_state=42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.models = {}
        self.histories = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_data(self, image_paths, labels, test_size=0.2, validation_split=0.2):
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        self.num_classes = len(le.classes_)
        
        X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=test_size, 
                                                            random_state=self.random_state, stratify=labels_encoded)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, 
                                                          random_state=self.random_state, stratify=y_train)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_loaders(self, data, batch_size=32, num_workers=4, model_type='classification'):
        if model_type == 'yolo':  # Custom YOLOv5 augmentation strategy
            train_transform = A.Compose([
                A.RandomResizedCrop(height=self.input_shape[0], width=self.input_shape[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.2),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(p=0.2),
                    A.MotionBlur(p=0.2),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:  # Standard classification augmentations
            train_transform = A.Compose([
                A.RandomResizedCrop(height=self.input_shape[0], width=self.input_shape[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.GaussNoise(p=0.2),
                    A.GaussianBlur(p=0.2),
                    A.MotionBlur(p=0.2),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        val_test_transform = A.Compose([
            A.Resize(height=self.input_shape[0], width=self.input_shape[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        train_dataset = ImageDataset(data['train']['image_paths'], data['train']['labels'], transform=train_transform)
        val_dataset = ImageDataset(data['val']['image_paths'], data['val']['labels'], transform=val_test_transform)
        test_dataset = ImageDataset(data['test']['image_paths'], data['test']['labels'], transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        return train_loader, val_loader, test_loader

    def create_model(self, model_name, pretrained=True):
        if model_name == 'yolov10':
            # Here we load YOLOv10 model
            print("Loading YOLOv10 model, featuring NMS-free training and an efficiency-accuracy driven design.")
            model = YOLO('yolov10.pt')  # Replace with correct model path once available
            model.to(self.device)
        else:
            # Load classification models
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=self.num_classes)
            model = model.to(self.device)
        
        return model

    def train_model(self, model, train_loader, val_loader, epochs=50, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')

        return model, {'train_loss': train_losses, 'val_loss': val_losses}

    def evaluate_model(self, model, test_loader):
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def train_and_evaluate(self, train_loader, val_loader, test_loader, models_to_train):
        for model_name in models_to_train:
            print(f"Training and evaluating {model_name}...")
            model = self.create_model(model_name)
            model, history = self.train_model(model, train_loader, val_loader)
            
            self.models[model_name] = model
            self.histories[model_name] = history
            
            results = self.evaluate_model(model, test_loader)
            self.results[model_name] = results
            
            if results['accuracy'] > self.best_score:
                self.best_score = results['accuracy']
                self.best_model = model

        return self.results

    def plot_training_history(self, model_name):
        history = self.histories[model_name]
        plt.figure(figsize=(12, 4))
        
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_results(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        results_df = pd.DataFrame({model: [self.results[model][metric] for metric in metrics] 
                                   for model in self.results.keys()}, index=metrics)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(results_df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Model Comparison')
        plt.show()

    def plot_confusion_matrix(self, model_name):
        cm = self.results[model_name]['confusion_matrix']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def load_image_data(data_source, split=True, test_size=0.2, val_size=0.2, random_state=42):
    if isinstance(data_source, str):
        if os.path.isfile(data_source) and data_source.endswith('.csv'):
            # Load from CSV file
            df = pd.read_csv(data_source)
            image_paths = df['image_path'].tolist()
            labels = df['label'].tolist()
        elif os.path.isdir(data_source):
            # Load from directory structure
            image_paths = []
            labels = []
            for class_name in os.listdir(data_source):
                class_dir = os.path.join(data_source, class_name)
                if os.path.isdir(class_dir):
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        image_paths.append(image_path)
                        labels.append(class_name)
        else:
            raise ValueError("Invalid data source. Please provide a CSV file or a directory.")
    elif isinstance(data_source, pd.DataFrame):
        # Data is already in a DataFrame
        image_paths = data_source['image_path'].tolist()
        labels = data_source['label'].tolist()
    else:
        raise ValueError("Invalid data source. Please provide a string path or a pandas DataFrame.")

    if split:
        # Split the data into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Split the train+val set into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        
        return {
            'train': {'image_paths': X_train, 'labels': y_train},
            'val': {'image_paths': X_val, 'labels': y_val},
            'test': {'image_paths': X_test, 'labels': y_test}
        }
    else:
        return {'image_paths': image_paths, 'labels': labels}

# Example usage
if __name__ == "__main__":
    # Load your image data here
    # image_paths should be a list of file paths to your images
    # labels should be a list of corresponding labels
    # image_paths, labels = load_image_data()
    
    # Initialize the classifier
    clf = AdvancedImageClassifier(input_shape=(224, 224, 3))
    
    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = clf.preprocess_data(image_paths, labels)
    
    # Create data loaders
    train_loader, val_loader, test_loader = clf.create_data_loaders({
        'train': {'image_paths': X_train, 'labels': y_train},
        'val': {'image_paths': X_val, 'labels': y_val},
        'test': {'image_paths': X_test, 'labels': y_test}
    })

    # Define the models you want to train, including YOLOv10
    models_to_train = [
        'resnet50',
        'efficientnet_b0',
        'vit_base_patch16_224',
        'deit_base_patch16_224',
        'swin_base_patch4_window7_224',
        'convnext_base',
        'yolov10' 
    ]
    
    # Train and evaluate all models
    results = clf.train_and_evaluate(train_loader, val_loader, test_loader, models_to_train)
    
    # Plot results
    clf.plot_results()
    
    # Plot training history for each model
    for model_name in clf.models.keys():
        clf.plot_training_history(model_name)
    
    # Plot confusion matrix for the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    clf.plot_confusion_matrix(best_model_name)
    
    # Print detailed results
    for model_name, result in results.items():
        print(f"\nResults for {model_name}:")
        for metric, value in result.items():
            if metric != 'confusion_matrix':
                print(f"{metric}: {value:.4f}")

    print(f"\nBest model: {best_model_name} with accuracy {clf.best_score:.4f}")

    # YOLOv10 specific data loaders and training
    train_loader, val_loader, test_loader = clf.create_data_loaders(
        {'train': {'image_paths': X_train, 'labels': y_train}, 'val': {'image_paths': X_val, 'labels': y_val}, 'test': {'image_paths': X_test, 'labels': y_test}},
        model_type='yolo'
    )
    
    # Train YOLOv10
    model = clf.create_model('yolov10')
    model, history = clf.train_model(model, train_loader, val_loader)

    # Evaluate YOLOv10
    results = clf.evaluate_model(model, test_loader)
    print(results)