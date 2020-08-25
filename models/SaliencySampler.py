from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)

model = model[0:10]
print(model.summary())
