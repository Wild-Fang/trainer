import turicreate as tc

# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('../dataset/Training', with_path=True)

# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: path[1:])

# Save the data for future use
data.save('training.sframe')

# Loading test data
test = tc.image_analysis.load_images('../dataset/Test', with_path=True)

test['label'] = test['path'].apply(lambda path: path[1:])

test.save('test.sframe')

# Explore interactively
data.explore()
