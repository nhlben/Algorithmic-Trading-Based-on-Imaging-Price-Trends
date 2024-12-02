from __init__ import *
import model
from plot_history import plot_history

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7600)
np.random.seed(7600)

# Class for early stop epoch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, num_epoch, loss_fn, optimizer, train_dataloader, val_dataloader, early_stopper, save_path):

    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    min_val_loss = 10
    for epoch in range(1, num_epoch+1):
        train_loss, train_correct, val_loss, val_correct = 0, 0, 0, 0
        # train
        model.train()
        for index, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)

            # clean grad
            optimizer.zero_grad()
            # forward
            output = model(data)
            # calculate loss
            loss = loss_fn(output, label)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            # record train loss
            train_loss += loss.item()
            # record correct identified label
            train_correct += (output.argmax(1) == label).sum()

        
        # evaluate
        model.eval()
        with torch.no_grad():
            for index, (data, label) in enumerate(val_dataloader):
                data = data.to(device)
                label = label.to(device)

                # forward
                output = model(data)
                # calculate loss
                loss = loss_fn(output, label)
                # record val loss
                val_loss += loss.item()
                # record correct identified label
                val_correct += (output.argmax(1) == label).sum()

        # Calculate average loss
        train_loss = train_loss/len(train_dataloader.sampler)
        train_loss_list.append(train_loss)
        
        train_accuracy = train_correct/len(train_dataloader.sampler)
        train_accuracy_list.append(train_accuracy.item())

        val_loss = val_loss/len(val_dataloader.sampler)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model
        val_loss_list.append(val_loss)
        val_accuracy = val_correct/len(val_dataloader.sampler)
        val_accuracy_list.append(val_accuracy.item())

        # print epoch info
        print(f"Epoch {epoch} / {num_epoch}:   Training Loss: {train_loss:.6f}   Training Accuracy: {train_accuracy:.4f}  Validation Loss: {val_loss:.6f}   Validation Accuracy: {val_accuracy:.4f} "
              , flush=True)

        # Early Stopper if val loss is not improved
        if early_stopper.early_stop(val_loss):  
            os.makedirs(save_path, exist_ok=True)
            print(f"The training is stopped early for no improvement in validation loss", flush=True) 
            print(f"The best model is saved at {save_path}", flush=True)
            torch.save(best_model, f"{save_path}/model.pt")
            return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, epoch

    os.makedirs(save_path, exist_ok=True)
    torch.save(best_model, f"{save_path}/model.pt")
    print(f"The best model is saved at {save_path}", flush=True)
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='number of classes')
    parser.add_argument('--model', type=str, default='I5', help='model name')
    parser.add_argument('--data_path', type=str, default='./data/I5R5/train', help='path for storing data')
    parser.add_argument('--save_path', type=str, default='./models/I5R5', help='path for storing model')
    parser.add_argument('--model_path', type=str, default='NONE', help='path for loading model')
    parser.add_argument('--early_stop_epoch', type=int, default=5, help='patience level for early stopper')
    parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer')
    opt = parser.parse_args()
    print(opt, flush=True)

    if opt.model_path != "NONE":
        # load model
        model = torch.load(f"{opt.model_path}/model.pt")
    else:
        # initialize model
        if opt.model == "I5":
            model = model.CNN_I5(opt.num_classes)
        elif opt.model == "I5_B3":
            model = model.CNN_I5_B3(opt.num_classes)
        elif opt.model == "DENSENET":
            model = models.densenet121(num_classes=opt.num_classes)
            model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif opt.model == "EFFICIENTNET":
            model = models.efficientnet_b0(num_classes=opt.num_classes)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride = (2, 2), padding = (1,1), bias=False)
        elif opt.model == "EFFICIENTNET_B4":
            model = models.efficientnet_b4(num_classes=opt.num_classes)
            model.features[0][0] = nn.Conv2d(1, 48, kernel_size=(3, 3), stride = (2, 2), padding = (1,1), bias=False)
        elif opt.model == "VIT":
            model = models.vit_b_16(num_classes=opt.num_classes, image_size=64)
            model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        elif opt.model == "VIT_I20":
            model = models.vit_b_16(num_classes=opt.num_classes, image_size=128)
            model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        elif opt.model == "VIT_MACD":
            model = models.vit_b_16(num_classes=opt.num_classes, image_size=96)
            model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        elif opt.model == "I20":
            model = model.CNN_I20(opt.num_classes)
    model.to(device)
    print(f"Device:{device}", flush=True)
    print(model, flush=True)

    # read dataset & apply transform
    if opt.model == "VIT":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((64,64)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])   
    elif opt.model == "VIT_MACD":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((96,96)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    elif opt.model == "VIT_I20":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((128,128)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])   
    else:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    dataset = datasets.ImageFolder(opt.data_path, transform=transform)
    print(f"Number of images: {len(dataset)}", flush=True)
    print(dataset.class_to_idx, flush=True)
    
    # divide dataset into train and val
    proportions = [0.7, 0.3]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    generator = torch.Generator().manual_seed(7600)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths, generator)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=True)
    print(f"training size:{len(train_dataloader.dataset)}, validation size:{len(val_dataloader.dataset)}", flush=True)

    # initialize loss function & optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    if opt.optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate , weight_decay=0.1)
    elif opt.optimizer =="ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate , weight_decay=0.01)
    elif opt.optimizer =="SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate , weight_decay=0.01)
    elif opt.optimizer =="RMSPROP":
        optimizer = optim.RMSprop(model.parameters(), lr=opt.learning_rate , weight_decay=0.01)
    # early stopperg
    early_stopper = EarlyStopper(patience=opt.early_stop_epoch, min_delta=5e-4)  # 5e-6 for real training

    # train model
    train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list, epoch = train(model, opt.num_epoch, loss_fn, optimizer, train_dataloader, 
                                                                                   val_dataloader, early_stopper, opt.save_path)
    
    # save df csv
    df = pd.DataFrame({"train_loss":train_loss_list, "train_acc":train_accuracy_list,
                "val_loss":val_loss_list, "val_acc":val_accuracy_list})
    df.to_csv(f"{opt.save_path}/training_history.csv", index=False)
    print(f"Training History saved", flush=True)

    # plot history
    plot_history(opt.save_path, epoch)


