from llava.train.sparse_train import train

if __name__ == "__main__":
    train(attn_implementation="sdpa")
