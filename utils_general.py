from mask_component_bi import MaskedModule
from utils_libs import *
from utils_dataset import *
from utils_models import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time

max_norm = 10


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, device, w_decay=None):
    acc_overall = 0;
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(2000, data_x.shape[0])
    n_test = data_x.shape[0]
    test_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    model.eval()

    model = model.to(device)
    with torch.no_grad():
        test_gen_iter = test_gen.__iter__()
        for i in range(int(np.ceil(n_test / batch_size))):
            batch_x, batch_y = test_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_test
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_test


# --- Helper functions

def avg_models(mdl, client_models, weight_list):
    n_node = len(client_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(client_models[i].named_parameters()))

    param_0 = client_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def set_client_from_params(mdl, params,device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            if ".p" not in name:
                temp = param.data.cpu().numpy().reshape(-1)
                param_mat[i, idx:idx + len(temp)] = temp
                idx += len(temp)
    return np.copy(param_mat)


# --- Training models fedavg

def train_model(model, train_x, train_y, test_x, test_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step, sch_gamma , device, print_verbose=True):
    n_train = train_x.shape[0]
    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                                shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put test_x=False if no test data given
    print_test = not isinstance(test_x, bool)

    model.train()

    for e in range(epoch):
        if print_verbose and (e == 0 or (e + 1) % print_per == 0):
            loss_train, acc_train = get_acc_loss(train_x, train_y, model, dataset_name, device, weight_decay)
            if print_test:
                loss_test, acc_test = get_acc_loss(test_x, test_y, model, dataset_name,device, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_train, loss_train, acc_test, loss_test, scheduler.get_lr()[0]),flush=True)
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_train, loss_train, scheduler.get_lr()[0]),flush=True)
            model.train()
        # Training
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model




def train_model_apfl(model, train_x, train_y, test_x, test_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step, sch_gamma , device, v, alpha_apfl=0.5, print_verbose=True):
    n_train = train_x.shape[0]
    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                                shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put test_x=False if no test data given
    print_test = not isinstance(test_x, bool)

    model.train()

    for e in range(epoch):
        if print_verbose and (e == 0 or (e + 1) % print_per == 0):
            loss_train, acc_train = get_acc_loss(train_x, train_y, model, dataset_name, device, weight_decay)
            if print_test:
                loss_test, acc_test = get_acc_loss(test_x, test_y, model, dataset_name,device, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_train, loss_train, acc_test, loss_test, scheduler.get_lr()[0]),flush=True)
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_train, loss_train, scheduler.get_lr()[0]),flush=True)
            model.train()
        # Training
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):

            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wt = copy.deepcopy(model.state_dict())


            v_bar = {}
            for k in model.state_dict().keys():
                v_bar[k] = alpha_apfl *  v[k].to(device) + (1-alpha_apfl) * wt[k]

            model.load_state_dict(v_bar)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = alpha_apfl* loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            for name, param in  model.named_parameters():
                param.data = v[name].data.clone().to(device)
            optimizer.step()
            v = copy.deepcopy(model.state_dict())
            model.load_state_dict(wt)

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    # to cpu
    for name in v:
        v[name] = v[name].to("cpu")
    return model, v

def train_scaffold_mdl(model, model_func, state_params_diff, train_x, train_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma, device, print_verbose=False):
    n_train = train_x.shape[0]

    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                                shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_train / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    step_loss = 0
    n_data_step = 0
    for e in range(epoch):
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            # Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
            #                                max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

        if print_verbose and (e == 0 or (e + 1) % print_per) == 0:
            step_loss /= n_data_step
            # if weight_decay != None:
            #     # Add L2 loss to complete f_i
            #     params = get_mdl_params([model], n_par)
            #     step_loss += (weight_decay)/2 * np.sum(params * params)

            print("Step %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, step_loss, scheduler.get_lr()[0]))
            step_loss = 0;
            n_data_step = 0

        model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model




def switch_training_mode(model, mode):
    for module in mask_modules(model):
        module.set_train_status(mode)

def mask_modules(model):
    new_modules = []
    for module in model.modules():
        if isinstance(module, MaskedModule):
            new_modules.append(module)
    return new_modules



def finetune_model(model, train_x, train_y,
                   learning_rate, batch_size, epoch, print_per, dataset_name, device, org_weights, weight_decay=1e-3, se_threshold=0, finetune_proximal=0,  print_verbose=True, mode= "LR+S"):
    gpu_org_weights ={}
    for name in org_weights:
        gpu_org_weights[name] = org_weights[name].to(device)

    n_train = train_x.shape[0]

    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                                shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model.train()
    # switch_training_mode(model, "train_p")
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            # print(batch_y)
            ## Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]
            loss = loss_f_i
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if ".p" in name and param.requires_grad:
                    reg = se_threshold
                    strength = reg * learning_rate
                    param.data = (param.data - strength) * (param.data >= strength) + (param.data + strength) * (
                        param.data <= -strength)


            for name, param in model.named_parameters():
                if ".p" not in name and param.requires_grad:

                    model.state_dict()[name] -= learning_rate * finetune_proximal * (
                                model.state_dict()[name] - gpu_org_weights[name] )

            epoch_loss += loss.item() * list(batch_y.size())[0]
        if print_verbose and (e) % print_per == 0:
            sum_number = 0.1
            non_zero = 0
            sum_number2 = 0.1
            non_zero2 = 0
            for name, param in model.named_parameters():
                if ".p" in name:
                    sum_number += torch.numel(param)
                    non_zero += torch.count_nonzero(param)
                    # print("name{} grad:{}".format(name, torch.count_nonzero(param.grad)/torch.numel(param.grad)) )
            loss_train, acc_train = get_acc_loss(train_x, train_y, model, dataset_name, device, 0)
            print(
                "Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f, p_Density %.6f, w_density %.6f, p-Norm %.6f  w-Norm %.6f" % (
                    e + 1, acc_train, loss_train, learning_rate, non_zero / sum_number, non_zero2/sum_number2,
                    sum([torch.norm(param.data)** 2 for name, param in model.named_parameters() if ".p" in name and "fc" not in name and "linear" not in name and "bn" not in name and "shortcut.1" not in name ]),
                    sum([torch.norm(param.data) ** 2 for name, param in model.named_parameters() if ".w" in name])),
                flush=True)
            model.train()
    return model




def train_model_fedSLR(model, model_func, alpha_coef, concat_w, concat_mu, train_x, train_y,
                   learning_rate, batch_size, epoch, print_per,
                   weight_decay, dataset_name, sch_step, sch_gamma,se_threshold, device, client_num,  print_verbose=True, mode=None):
    n_train = train_x.shape[0]

    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                                shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    start_time1 = time.time()

    concat_w = concat_w.to(device)
    concat_mu = concat_mu.to(device)
    model.train()
    model.to(device)
    switch_training_mode(model, "only_w")
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                weight_decay= alpha_coef+weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)


    for e in range(epoch):
        # Training
        epoch_loss = 0
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            # print(model)
            # print(batch_y.reshape(-1).long(), flush=True)
            # print(y_pred.shape)
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            local_par_list = None
            for name, param in model.named_parameters():
                if ".w" in name:
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                # print(torch.norm(local_par_list), flush=True)
            loss =  loss_f_i +   torch.sum(local_par_list * (-alpha_coef*concat_w - concat_mu))

            # loss = loss_f_i+weight_decay/2*torch.norm(local_par_list)**2
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= int(np.ceil(n_train / batch_size))

        if print_verbose and (e) % print_per == 0:
            sum_number = 0
            non_zero = 0
            for name, param in model.named_parameters():
                if ".p" in name:
                    sum_number += torch.numel(param)
                    non_zero += torch.count_nonzero(param)
        model.train()
        scheduler.step()
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
    end_time1 = time.time()
    time1 = end_time1 - start_time1
    print("elapse time {}".format(time1),flush=True)
    return model
