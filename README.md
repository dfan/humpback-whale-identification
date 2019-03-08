# humpback-whale-identification
(Kaggle competition)[https://www.kaggle.com/c/humpback-whale-identification/] to identify humpback whales from just their tail. This was my first Kaggle competition ever and I wanted to explore transfer learning. The idea of transfer learning is starting with a pretrained model, and finetuning it to do well on other tasks. I decided to use ResNet50 pretrained on ImageNet, since ImageNet is the standard benchmark dataset for object recognition and ResNet achieves state-of-art performance.

### Evaluation
This Kaggle competition evaluates on Mean Average Precision (MAP); one is allowed to make up to 5 predictions per image (in decreasing order of confidence), and MAP is higher if the ground truth label comes earlier in the predicted labels. My initial MAP score was quite low (~0.20 / 5th percentile), but the following adjustments boosted my MAP to ~0.65 / 50th percentile.

### Things that Helped

- Using training and test-time augmentation (adding transformations to further increase the diversity of images the classifier sees, and avoid overfitting).
- Add batch normalization and dropout before the final fully-connected layer.
- Normalize input images by the ImageNet mean and standard deviation vectors (was unaware of this standard practice).
- Cropping the input images by computing bounding boxes. I used a pretrained Keras model from a public Kaggle kernel to crop the images (see `crop_images.py`).
- Training the entire model instead of just the fully-connected layer. I use a trick where the ResNet backbone is frozen initially to help the fully-connected layer weights stabilize, and then I unfreeze all layers and train the entire model.

### Things that Did Not Help
The following adjustments did not help by a noticeable amount.

- Ensemble models: I tried ensembling ResNet50 with ResNet101 and ResNet50 with Resnet34, but this didn't improve my performance. Ensemble models usually win all Kaggle competitions, so I suspect my failure here was due to either improper model selection, or inference. I naively took the average of the probabilities from my two networks, but perhaps a weighted average would make more sense. ResNet50 and ResNet101/34 learn essentially the same features, so maybe ResNet50 + Inception would make more sense, for example.
- Training in RGB - grayscale images have strictly less information than RGB images, so I thought that training in RGB would yield benefits. I did not experience a noticable improvement in MAP, but inference took longer when loading RGB images.
- Removing new_whales. My final code removes new_whales anyway from the training set, and uses probabilistic inference to guess new_whales, but I didn't experience a significant gain by doing this. Kaggle discussion threads suggested that omitting new_whales would help, since there are significantly more new_whales than some of the known whale classes. I didn't experience any issues with class imbalance though.

### Future Work
Some of the best entries used Siamese networks. The idea is to train a network to **differentiate** between inputs, rather than classify inputs. Siamese networks consist of two identical networks, each taking an input image. The loss function is contrastive. I would try playing with Siamese networks, as well as being smarter about implementing ensemble models.

### Takeaways
This was my first Kaggle competition, and I really enjoyed it. I got to try a lot of practical training tricks that I had previously only read about (like dealing with class imbalance, test-time augmentation, ensemble models). It's pretty hard to do well in Kaggle - I spent a couple of full days after my winter finals on this competition, and only ended up around 50th percentile. This was despite having access to free GPUs from university; most Kagglers probably pay for AWS or GCP instances.
