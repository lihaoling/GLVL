import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util
import test
import parser_joint
import datasets_ws
from model import net
from model.sync_batchnorm import convert_model
import yaml
import importlib
# from SuperPoint_fronted import SuperPoint_fronted
import faulthandler

faulthandler.enable()
torch.backends.cudnn.benchmark = True  # Provides a speedup

#### Initial setup: parser, logging...
args = parser_joint.parse_arguments()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

torch.set_default_tensor_type(torch.FloatTensor)
start_time = datetime.now()

args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
print(f"Train_joint Arguments: {args}")
print(f"SuperPointNet training Arguments: {config}")
print(f"The outputs are being saved in {args.save_dir}")
print(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
print(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train",
                                          args.negs_num_per_query)
print(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
print(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
print(f"Test set: {test_ds}")

spt_data = datasets_ws.dataLoader(config, dataset=config['data']['dataset'], warp_input=True)
train_loader, val_loader = spt_data['train_loader'], spt_data['val_loader']


#### Initialize model
model = net.GeoLocalizationNet(args)
model = model.to(args.device)


best_model_state_dict = torch.load(join('/home/lhl/data/data/visgeoloca/logs/default/2023-08-10_09-05-21/checkpoints', "superPointNet_85000_checkpoint.pth.tar"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

model = torch.nn.DataParallel(model)


mod = importlib.import_module(config['front_end_model'])
train_model_frontend = getattr(mod, config['front_end_model'])
train_agent = train_model_frontend(config, save_path=args.save_dir, device=args.device)

train_agent.loadModel(model=model)
#### Setup Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

best_r5 = start_epoch_num = not_improved_num = 0
if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

n_iter = 0
#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    print(f"Start training epoch: {epoch_num:02d}")

    ###############################################################################################
    ######################################### retrieval ###########################################
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        print(f"Cache: {loop_num} / {loops_num}")

        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False

        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)

        model = model.train()

        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):


            # Compute features of all images (images contains queries, positives and negatives)
            features = model(images.to(args.device), 'retrieval')
            loss_triplet = 0

            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                loss_triplet += criterion_triplet(features[queries_indexes],
                                                  features[positives_indexes],
                                                  features[negatives_indexes])

            del features
            loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet

        # Visualization of Epoch loss
        print(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
              f"current batch triplet loss = {batch_loss:.4f}, " +
              f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        epoch_loss_log_dir = join(args.save_dir, "epoch_loss")
        train_writer = SummaryWriter(log_dir=epoch_loss_log_dir)
        train_writer.add_scalar('Mean Epoch Loss', epoch_losses.mean(), epoch_num)

    print(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
          f"average epoch triplet loss = {epoch_losses.mean():.4f}")

    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    print(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    util.retrieval_save_checkpoint(args, {
        "epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")

    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        print(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        print(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        if not_improved_num >= args.patience:
            print(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break




    ##################################################################################################
    ######################################## SuperPointNet ###########################################

    running_losses = []
    for j in range(5):
        print(f"The {j}-th SuperPointNet training Process")
        for i, sample_train in tqdm(enumerate(train_loader)):
            n_iter = n_iter + 1
            # train one sample
            loss_out = train_agent.train_val_sample(sample_train, n_iter=i, train=True)
            running_losses.append(loss_out)
            # run validation
            if n_iter % config["validation_interval"] == 0:
                logging.info("====== Validating...")
                for j, sample_val in enumerate(val_loader):
                    train_agent.train_val_sample(sample_val, n_iter=n_iter + j, train=False)
                    if j > config.get("validation_size", 3):
                        break
            # save model
            if n_iter % config["save_interval"] == 0:
                logging.info(
                    "save model: every %d interval, current iteration: %d",
                    config["save_interval"],
                    n_iter,
                )
                train_agent.saveModel(n_iter=n_iter)
            # ending condition
            if n_iter > config['train_iter']:
                # end training
                logging.info("End training: %d", n_iter)
                break

print(f"Best R@5: {best_r5:.1f}")
print(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
print(f"Recalls on {test_ds}: {recalls_str}")
