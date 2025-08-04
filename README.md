# EnVGG16
🎵 EnVGG16: Enhanced VGG16 for Music Genre Classification

EnVGG16 is a deep learning model built upon the classic VGG16 architecture, tailored specifically for the task of music genre classification using Mel-spectrograms as input. It leverages the power of transfer learning and introduces architectural enhancements to significantly improve classification accuracy and generalization.

📌 Key Features:

🎶 Designed for music genre classification using the GTZAN dataset (10 genres, 1000 clips).
🎨 Input data consists of RGB spectrograms, resized and normalized in the style of ImageNet preprocessing.
🧠 Uses VGG16 pretrained on ImageNet for feature extraction, with custom layers added for task-specific learning.
⚙️ Fine-tuning applied to the last 8 layers of VGG16 for domain adaptation.
💡 Integrates batch normalization, dropout, and AdamW optimizer with early stopping for efficient training.
📈 Achieved a classification accuracy of 97.43%, outperforming ResNet34, ResNet50, AlexNet, and baseline VGG16.

📊 Results (on GTZAN dataset):

Accuracy:	97.43%
Macro F1-Score: 0.98
Per-class F1: Most genres achieved F1 = 1.00
Lowest performance: Blues (F1 = 0.86), Classical (F1 = 0.93)
