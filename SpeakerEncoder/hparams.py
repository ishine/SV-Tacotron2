import os


# Audio
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
tisv_frame = 180
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

# Model
n_mels_channel = 80
hidden_dim = 768
num_layer = 3
speaker_dim = 256
re_num = 1e-6

# Train
dataset_path = "./dataset"
dataset_test_path = "./dataset_test"
origin_data = "VCTK-Corpus-Processed"
total_utterance = 200
N = 20  # batch_size
M = 10
learning_rate = 0.01
epochs = 500
checkpoint_path = os.path.join("SpeakerEncoder", "model")
save_step = 100
log_step = 5
clear_Time = 20
