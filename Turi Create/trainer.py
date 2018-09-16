import turicreate as tc

# Load the data
train_data =  tc.SFrame('training.sframe')
test_data = tc.SFrame('test.sframe')

# Create the model
model = tc.image_classifier.create(train_data, target='label')

# Save predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('../TuriCreate.model')

# Export for use in Core ML
model.export_coreml('AnimalsTuri.mlmodel')
