class Config:
    MODEL_NAME = "google/flan-t5-base"
    SEED = 42
    MAX_INPUT_LENGTH = 64
    MAX_OUTPUT_LENGTH = 10
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    LR = 5e-5
    EPOCHS = 10
    OUTPUT_DIR = "./flan_t5_action_predictor"
    LOGGING_STEPS = 50
    DATA_FILE = "../data/clean_data.json"
