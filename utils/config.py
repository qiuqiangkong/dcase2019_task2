sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size
logmel_eps = -100   # This value indicates silence used for padding

labels = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 
    'Applause', 'Bark', 'Bass_drum', 'Bass_guitar', 
    'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation', 
    'Bus', 'Buzz', 'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 
    'Child_speech_and_kid_speaking', 'Chink_and_clink', 'Chirp_and_tweet', 
    'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket', 
    'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 
    'Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip', 
    'Electric_guitar', 'Fart', 'Female_singing', 
    'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 
    'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 
    'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock', 
    'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 
    'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 
    'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run', 'Scissors', 
    'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 
    'Slam', 'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 
    'Toilet_flush', 'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 
    'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf', 
    'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']
    
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)

folds_num = 4