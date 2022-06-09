X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []

y_train_enc = []
train_input_len = []
y_train_len = []
y_valid_enc = []
valid_input_len = []
y_valid_len = []

images_path = "InkData_word_processed_resized"
labels_path = "InkData_word_labels"

# os.chdir("InkData_word_processed_resized")

label_files = os.listdir(labels_path)

with open("train_set.txt", "r") as file:
  for value in file:
    value = value.split(".")[0]
    
    for filename in label_files:
      filename = filename.split(".")[0]

      if value in filename:
        image = cv2.cvtColor(cv2.imread(os.path.join(images_path,filename+".png")),cv2.COLOR_BGR2GRAY)
        image = image/255.
        image = np.expand_dims(image, axis=2)
        X_train.append(image)
        
        with open(os.path.join(labels_path,filename+".txt"), "r") as label_file:
          for label in label_file:
            y_train.append(label)
            y_train_enc.append(encode_to_labels(label))
            y_train_len.append(len(label))
            train_input_len.append(31)

with open("validation_set.txt", "r") as file:
  for value in file:
    value = value.split(".")[0]

    for filename in label_files:
      filename = filename.split(".")[0]

      if value in filename:
        image = cv2.cvtColor(cv2.imread(os.path.join(images_path,filename+".png")),cv2.COLOR_BGR2GRAY)
        image = image/255.
        image = np.expand_dims(image, axis=2)
        X_valid.append(image)

        with open(os.path.join(labels_path,filename+".txt"), "r") as label_file:
          for label in label_file:
            y_valid.append(label)
            y_valid_enc.append(encode_to_labels(label))
            y_valid_len.append(len(label))
            valid_input_len.append(31)

with open("test_set.txt", "r") as file:
  for value in file:
    value = value.split(".")[0]

    for filename in label_files:
      filename = filename.split(".")[0]

      if value in filename:
        image = cv2.cvtColor(cv2.imread(os.path.join(images_path,filename+".png")),cv2.COLOR_BGR2GRAY)
        image = image/255.
        image = np.expand_dims(image, axis=2)
        X_test.append(image)

        with open(os.path.join(labels_path,filename+".txt"), "r") as label_file:
          for label in label_file:
            y_test.append(label)
            max_len_label = max(max_len_label,len(label))