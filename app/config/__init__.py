class Config:

    checkpoint_path = 'app/checkpoints/model.pth.tar'
    vocabulary_path = "app/data/info.pkl"
    number_of_classes = 2000
    number_of_frames = 64
    stride = 8
    batch_size = 10
    topk = 1