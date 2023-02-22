from multiprocessing.spawn import freeze_support
from utils_methods import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], type=str, default='CIFAR100')
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--rule-arg', default=0.1, type=float)
    parser.add_argument('--act_prob', default=0.1, type=float)
    parser.add_argument('--method', choices=['FedAvg', 'SCAFFOLD',"FedSLR","PerFedAvg", "FedRep", "LgFedAvg","standalone","APFL"], type=str, default='FedSLR')
    parser.add_argument('--n_client', default=100, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--local_epochs', default=2, type=int)
    parser.add_argument('--alpha', default=0.05, type=float)
    parser.add_argument('--alpha_coef', default=0.1, type=float)
    parser.add_argument('--local_learning-rate', default=0.1, type=float)
    parser.add_argument('--global_learning-rate', default=1.0, type=float)
    parser.add_argument('--lr_decay', default=0.998, type=float)
    parser.add_argument('--sch_gamma', default=1, type=float)
    parser.add_argument('--test_per', default=5, type=int)
    parser.add_argument('--batchsize', default=20, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--sparse-threshold', default=0.001, type=float)
    parser.add_argument('--rank-threshold', default=0.0001, type=float)
    parser.add_argument('--finetune_proximal', default=0.1, type=float)
    parser.add_argument('--model', default="Resnet18", type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--mode', default="LR+S", type=str)
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # Dataset initialization
    data_path = './'

    n_client = args.n_client
    # Generate IID or Dirichlet distribution
    if args.non_iid is False:
        data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                                     data_path=data_path)
    else:
        data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='Drichlet',
                                     rule_arg=args.rule_arg, data_path=data_path)

    if args.dataset == 'CIFAR10':
        model_name = 'Resnet18'
    elif args.dataset == 'CIFAR100':
        model_name = 'Cifar100_Resnet18'
    elif args.dataset == "tinyimagenet":
        model_name = 'tinyimagenet_Resnet18'

    # Common hyperparameters
    com_amount = args.epochs
    save_period = 10000

    weight_decay = 1e-3
    print("weight decay {}".format(weight_decay))
    batch_size = args.batchsize
    act_prob = args.act_prob
    suffix = model_name
    lr_decay_per_round = args.lr_decay
    # Initialize the model for all methods with a random seed or load it from a saved initial model
    if args.non_iid:
        iid = "FALSE"
    else:
        iid = "TRUE"
    # Model function
    if args.method=="FedSLR" :
         model_func = lambda: construct_mask_model(client_model(model_name, args=args), args)

    else:
        model_func = lambda: client_model(model_name, args=args)

    init_model = model_func()
    print(sum([torch.numel(param) for name, param in init_model.named_parameters()]))

    if args.method =="FedSLR":
        epoch = args.local_epochs
        alpha_coef = args.alpha_coef
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_fedSLR(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                     batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                     weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                     alpha_coef=alpha_coef, sch_step=1, sch_gamma=args.sch_gamma,
                     rand_seed=0, lr_decay_per_round=lr_decay_per_round,device=device , rank_threshold= args.rank_threshold ,sparse_threshold= args.sparse_threshold, iid= iid, model_name =args.model, mode= args.mode)

    if args.method == "APFL":
        epoch = args.local_epochs
        alpha_coef = args.alpha_coef
        learning_rate = args.local_learning_rate
        test_per = args.test_per
        train_apfl(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                    batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                    weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                    sch_step=1, sch_gamma=args.sch_gamma,lr_decay_per_round=lr_decay_per_round,device=device, iid= iid)

    if args.method == "standalone":
        epoch = args.local_epochs
        alpha_coef = args.alpha_coef
        learning_rate = args.local_learning_rate
        test_per = args.test_per
        train_standalone(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                    batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                    weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                    sch_step=1, sch_gamma=args.sch_gamma, lr_decay_per_round=lr_decay_per_round,device=device, iid= iid)

    if args.method == "FedRep":
        learning_rate = args.local_learning_rate
        epoch = args.local_epochs
        test_per = args.test_per
        train_FedRep(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                         batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                         weight_decay=weight_decay, model_func=model_func, init_model=init_model, device=device,
                         lr_decay_per_round=lr_decay_per_round, rand_seed=0,sch_step=1, sch_gamma=1, iid= iid)


    if args.method == "LgFedAvg":
        learning_rate = args.local_learning_rate
        epoch = args.local_epochs
        test_per = args.test_per
        train_LGFedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                     batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                     weight_decay=weight_decay, model_func=model_func, init_model=init_model, device=device,
                     lr_decay_per_round=lr_decay_per_round, rand_seed=0, sch_step=1, sch_gamma=1, iid= iid)



    elif args.method == 'FedAvg':
        epoch = args.local_epochs
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_FedAvg(data_obj=data_obj,act_prob=act_prob, learning_rate=learning_rate,
                     batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                     weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                     sch_step=1, sch_gamma=1, rand_seed=0, device=device,lr_decay_per_round=lr_decay_per_round, iid= iid)


    elif args.method == 'SCAFFOLD':
        epoch = args.local_epochs

        n_data_per_client = np.concatenate(data_obj.client_x, axis=0).shape[0] / n_client
        n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)

        n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob,
            learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, n_minibatch=n_minibatch,
            com_amount=com_amount, test_per=test_per, weight_decay=weight_decay,
            model_func=model_func, init_model=init_model,finetune_proximal=args.finetune_proximal,
            sch_step=1, sch_gamma=args.sch_gamma, rand_seed=0, device=device,lr_decay_per_round=lr_decay_per_round, iid= iid)


