import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Define constants
INPUT_FEATURES = 100  # Number of features for input sequences
OUTPUT_FEATURES = 200  # Number of features for output sequences
BATCH_SIZE = 64
EPOCHS = 50

# Sample data (you would collect this from users in a real scenario)
input_texts = ['what is the path name ?', 'New path name ?', 'Would you like to create a new path ?', 'Please provide full path name.']
target_texts = ['\\storage_name\share_name', '/local/mnt/path', '/new_path']

# Tokenize the input and output texts
input_tokens = set()
output_tokens = set()

for input_text, target_text in zip(input_texts, target_texts):
    for char in input_text:
        if char not in input_tokens:
            input_tokens.add(char)
    for char in target_text:
        if char not in output_tokens:
            output_tokens.add(char)

input_tokens = sorted(list(input_tokens))
output_tokens = sorted(list(output_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(output_tokens)

# Create dictionaries to convert characters to indices and vice versa
input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
target_token_index = dict([(char, i) for i, char in enumerate(output_tokens)])

# Prepare data for model training
encoder_input_data = np.zeros((len(input_texts), INPUT_FEATURES, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), OUTPUT_FEATURES, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), OUTPUT_FEATURES, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# Define LSTM-based sequence-to-sequence model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(INPUT_FEATURES, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(OUTPUT_FEATURES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

# Inference mode (to test the model)
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(OUTPUT_FEATURES,))
decoder_state_input_c = Input(shape=(OUTPUT_FEATURES,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Function to decode sequence
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = output_tokens[0, -1, sampled_token_index]
        decoded_sentence += output_tokens[0, -1, sampled_token_index]
        if (sampled_char == '\n' or
           len(decoded_sentence) > OUTPUT_FEATURES):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence

# Example usage
input_seq = encoder_input_data[0:1]
decoded_sentence = decode_sequence(input_seq)
print('Decoded sentence:', decoded_sentence)
