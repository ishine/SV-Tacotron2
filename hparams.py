from text import symbols

# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Mel
n_mel_channels = 80
n_frames_per_step = 1

# Bigger
# # PostNet
# postnet_embedding_dim = 512
# postnet_kernel_size = 5
# postnet_n_convolutions = 5

# # PreNet
# prenet_dim = 256

# # Encoder
# encoder_n_convolutions = 3
# encoder_embedding_dim = 512
# encoder_kernel_size = 5

# # Decodr
# attention_rnn_dim = 1024
# decoder_rnn_dim = 1024
# max_decoder_steps = 1000
# gate_threshold = 0.5
# p_attention_dropout = 0.1
# p_decoder_dropout = 0.1
# attention_location_kernel_size = 31
# attention_location_n_filters = 32
# attention_dim = 128

# # Speaker Encoder
# tisv_frame = 180

# # Tacotron2
# mask_padding = True
# n_symbols = len(symbols)
# symbols_embedding_dim = 512

# Smaller
# PostNet
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

# PreNet
prenet_dim = 256

# Encoder
encoder_n_convolutions = 3
encoder_embedding_dim = 256
encoder_kernel_size = 5

# Decodr
attention_rnn_dim = 512
decoder_rnn_dim = 512
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1
attention_location_kernel_size = 31
attention_location_n_filters = 32
attention_dim = 128

# Speaker Encoder
tisv_frame = 180

# Tacotron2
mask_padding = True
n_symbols = len(symbols)
symbols_embedding_dim = 256

# Train
batch_size = 32
epochs = 300
dataset_path = "dataset"
vctk_path = "VCTK-Corpus-Processed"
learning_rate = 1e-3
weight_decay = 1e-6
checkpoint_path = "./model_new"
grad_clip_thresh = 1.0
save_step = 100
log_step = 5
clear_Time = 20
