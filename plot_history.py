from __init__ import *

def plot_history(save_path, epoch):
    df = pd.read_csv(f"{save_path}/training_history.csv")
    train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list = df["train_loss"], df["train_acc"], df["val_loss"], df["val_acc"]
    
    # plot loss
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, epoch+1, 10))
    plt.ylabel("CELoss")
    plt.title("Training and Validation Loss")
    plt.savefig(f"{save_path}/loss_history.png")
    print(f"Loss History saved", flush=True)
    plt.close()

    # plot accuracy
    plt.plot(train_accuracy_list, label="Training Accuracy")
    plt.plot(val_accuracy_list, label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.xticks(np.arange(0, epoch+1, 10))
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.savefig(f"{save_path}/accuracy_history.png")
    print(f"Accuracy History saved", flush=True)
    plt.close()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./data/I5R5/train', help='path for storing data')
    parser.add_argument('--num_epoch', type=int, default=50, help='epoch')   
    opt = parser.parse_args()
   
    plot_history(opt.save_path, opt.num_epoch)