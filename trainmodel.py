from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print('loading Actions')
for action in actions:
    print(f'loaded action - {action}')
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(1,sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

new_sequences = []
new_labels = []
for seq, label in zip(sequences, labels):
    if seq[0].any() == None:
        pass
    else:
        new_sequences.append(seq)
        new_labels.append(label)

X = np.array(new_sequences)

y = to_categorical(new_labels).astype(int)
print('Splitting The dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print('Create Model')
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(29,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

print('Compiling Model')
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print('Model Compiled')
print(model.summary())
model.fit(X_train, y_train, epochs=50)


model.save('model.h5')