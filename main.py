import time, os, sys, warnings, random
import logging, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from utils.evaluation.meter import AverageMeter
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# modify for wider dataset and vit models

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=100, type=int)

    parser.add_argument('--save_dir', default='./checkpoint/', type=str, help='save path')
    parser.add_argument('--seed', default=99, type=int, help='seed for initializing training. ')
    parser.add_argument('-s','--summary_writer', action='store_true',  default=False, help="start tensorboard")
    # gpus
    parser.add_argument("--gpus", default='0', type=str)

    
    args = parser.parse_args()
    return args


def build_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

    # args.save_dir
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    file_path = os.path.join(args.save_dir, '{}_{}.log.txt'.format(args.dataset, args.model))
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, args

# show config
def show_args(args, logger):
    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")

def train(i, args, model, train_loader, optimizer, warmup_scheduler, logger, meters):
    model.train()
    epoch_begin = time.time()
    for index, data in enumerate(train_loader):
        batch_begin = time.time() 
        img = data['img'].cuda()
        target = data['target'].cuda()

        optimizer.zero_grad()
        logit, loss = model(img, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t = time.time() - batch_begin

        if index % args.print_freq == 0:
            logger.info("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))
 
        meters['loss'].update(loss.item(), img.size(0))


        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()
    
    t = time.time() - epoch_begin
    logger.info("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader, test_file, logger):
    model.eval()
    logger.info("Test on Epoch {}".format(i))
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    # cal_mAP OP OR
    mAP = evaluation(result=result_list, types=args.dataset, ann_path=test_file[0], logger=logger) 
    return mAP



def main():
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Bulid Logger
    logger, args = build_logger(args)

    # Bulid Summary
    if args.summary_writer:
        summary_writer = SummaryWriter(log_dir=args.save_dir)

    # show config
    show_args(args, logger)
    
    # initialize_meters
    meters = {}
    meters['loss'] = AverageMeter('loss')

    if args.seed is not None:
        logger.info('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # model
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        
    model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # data
    if args.dataset == "voc07":
        train_file = ["data/voc07/trainval_voc07.json"]
        test_file = ['data/voc07/test_voc07.json']
        step_size = 4
    if args.dataset == "coco":
        train_file = ['data/coco/train_coco2014.json']
        test_file = ['data/coco/val_coco2014.json']
        step_size = 5
    if args.dataset == "wider":
        train_file = ['data/wider/trainval_wider.json']
        test_file = ["data/wider/test_wider.json"]
        step_size = 5
        args.train_aug = ["randomflip"]

    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset, logger)
    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset, logger)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        momentum=args.momentum, weight_decay=args.w_d)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    best_mAP = 0.0
    best_epoch = 0.0
    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer, warmup_scheduler, logger, meters)        
        mAP = val(i, args, model, test_loader, test_file, logger)
        is_best = mAP > best_mAP
        if is_best == True:
            best_mAP = max(mAP, best_mAP)
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_checkpoint.pth'))
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}.pth'.format(args.dataset, i)))
        logger.info("In {}-epoch, Best mAP:{:.5f}".format(best_epoch, best_mAP))

        if args.summary_writer:
            summary_writer.add_scalar('val_mAP', mAP, i)
            summary_writer.add_scalar('avg_val_loss', meters['loss'].average(), i)

        scheduler.step()

    if args.summary_writer:
        summary_writer.close()


if __name__ == "__main__":
    main()
