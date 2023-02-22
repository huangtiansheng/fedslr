from utils_general import *
import numpy as np
import torch


def record_npy(name, accuracy):
    np.save('results/' + name, accuracy)


def train_FedAvg(data_obj, act_prob, learning_rate, batch_size,
                 epoch, com_amount, test_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma, device, skip_training=False,
                 rand_seed=0, lr_decay_per_round=1, finetune_proximal=0.1, iid="FALSE"):
    n_client = data_obj.n_client

    client_x = data_obj.client_x;
    client_y = data_obj.client_y
    test_acc1 = []
    test_acc3 = []
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    client_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_client X n_par

    client_models = list(range(n_client))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
    ditto_params = [copy.deepcopy(cld_mdl_param) for i in range(n_client)]
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    if not skip_training:
        for i in range(com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clients = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clients) != 0:
                    break

            # print('Communication Round', i + 1, flush=True)
            # print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            del client_models
            client_models = list(range(n_client))
            for client in selected_clients:
                train_x = client_x[client]
                train_y = client_y[client]
                test_x = False
                test_y = False

                client_models[client] = model_func().to(device)
                client_models[client].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
                comm_param_num = 0
                for name, param in avg_model.named_parameters():
                    comm_param_num += torch.numel(param)
                # print("downlink_comm: {}".format(comm_param_num))
                # print("uplink_comm: {}".format(comm_param_num))
                # for params in client_models[client].parameters():
                #     params.requires_grad = True
                client_models[client] = train_model(client_models[client], train_x, train_y,
                                                    test_x, test_y,
                                                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch, 5,
                                                    weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma, device)

                client_params_list[client] = get_mdl_params([client_models[client]], n_par)[0]
                org_weights = {}
                for name, param in avg_model.named_parameters():
                    org_weights[name] = copy.deepcopy(param)
                client_models[client] = set_client_from_params(client_models[client].to(device), ditto_params[client],
                                                               device)
                # activate all layers
                for params in client_models[client].parameters():
                    params.requires_grad = True
                client_models[client] = finetune_model(client_models[client],
                                                       train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, 1, 1,
                                                       data_obj.dataset,
                                                       device,
                                                       finetune_proximal=finetune_proximal, org_weights=org_weights,
                                                       print_verbose=True)

                ditto_params[client] = get_mdl_params([client_models[client]], n_par)[0]
            avg_model = set_client_from_params(model_func(), np.mean(client_params_list[selected_clients], axis=0),
                                               device)
            # all_model = set_client_from_params(model_func(), np.mean(client_params_list, axis=0))

            if (i + 1) % test_per == 0:
                accs = []
                losses = []
                accs2 = []
                losses2 = []
                for idx in range(n_client):
                    loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                       avg_model, data_obj.dataset, device, 0)
                    accs.append(acc_test)
                    losses.append(loss_test)

                    ditto_model = set_client_from_params(model_func(), ditto_params[idx], device)
                    loss_test2, acc_test2 = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                         ditto_model, data_obj.dataset, device, 0)
                    accs2.append(acc_test2)
                    losses2.append(loss_test2)

                print("**** all  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
                print("**** ditto  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs2), np.mean(losses2)), flush=True)
                test_acc1.append(accs)
                test_acc3.append(accs2)

        file_name = "{}_fedavg_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc1)

        file_name = "{}_ditto_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc3)
        Path = "model/{}_avg_global_model".format(data_obj.dataset)
        torch.save({
            'model_state_dict': avg_model.state_dict(),
        }, Path)
    return


def train_FedRep(data_obj, act_prob, learning_rate, batch_size,
                 epoch, com_amount, test_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma, device, skip_training=False,
                 rand_seed=0, lr_decay_per_round=1, finetune_proximal=0.1, iid="FALSE"):
    n_client = data_obj.n_client

    client_x = data_obj.client_x;
    client_y = data_obj.client_y
    test_acc1 = []
    test_acc3 = []
    n_par = len(get_mdl_params([model_func()])[0])
    client_models = list(range(n_client))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    shared_layer = [name for name, param in avg_model.named_parameters() if "linear" not in name and "fc" not in name]
    g_param = {name: copy.deepcopy(param) for name, param in avg_model.named_parameters() if name in shared_layer}
    p_param = {name: copy.deepcopy(param) for name, param in avg_model.named_parameters() if name not in shared_layer}
    print([name for name in g_param])
    print([name for name in p_param])
    print(sum([torch.numel(g_param[name]) for name in g_param]) / (
                sum([torch.numel(g_param[name]) for name in g_param]) + sum(
            [torch.numel(p_param[name]) for name in p_param])))
    p_params = [copy.deepcopy(p_param) for i in range(n_client)]
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    if not skip_training:
        for i in range(com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clients = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clients) != 0:
                    break

            # print('Communication Round', i + 1, flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            del client_models
            client_models = list(range(n_client))
            g_param_temp = copy.deepcopy(g_param)
            g_param = {name: 0 for name in g_param}
            for client in selected_clients:
                train_x = client_x[client]
                train_y = client_y[client]
                test_x = False
                test_y = False

                client_models[client] = model_func().to(device)
                w_full = {}
                for name in g_param:
                    w_full[name] = g_param_temp[name]
                for name in p_param:
                    w_full[name] = p_params[client][name]
                client_models[client].load_state_dict(w_full)
                client_models[client] = train_model(client_models[client], train_x, train_y,
                                                    test_x, test_y,
                                                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch, 2,
                                                    weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma, device)
                for name in g_param:
                    g_param[name] += 1 / len(selected_clients) * copy.deepcopy(client_models[client].state_dict()[name])
                for name in p_params[0]:
                    p_params[client][name] = copy.deepcopy(client_models[client].state_dict()[name])

            if (i + 1) % test_per == 0:
                accs = []
                losses = []
                for idx in range(n_client):
                    temp_model = model_func().to(device)
                    w_full = {}
                    for name in g_param:
                        w_full[name] = g_param_temp[name]
                    for name in p_param:
                        w_full[name] = p_params[idx][name]
                    temp_model.load_state_dict(w_full)
                    loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                       temp_model, data_obj.dataset, device, 0)
                    accs.append(acc_test)
                    losses.append(loss_test)
                print("**** p  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
                test_acc1.append(accs)

        file_name = "{}_fedrep_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc1)
    return


def train_LGFedAvg(data_obj, act_prob, learning_rate, batch_size,
                   epoch, com_amount, test_per, weight_decay,
                   model_func, init_model, sch_step, sch_gamma, device, skip_training=False,
                   rand_seed=0, lr_decay_per_round=1, finetune_proximal=0.1, iid="FALSE"):
    n_client = data_obj.n_client

    client_x = data_obj.client_x;
    client_y = data_obj.client_y
    test_acc1 = []
    client_models = list(range(n_client))
    avg_model = model_func()
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    shared_layer = [name for name, param in avg_model.named_parameters() if "linear" in name or "fc" in name]
    g_param = {name: copy.deepcopy(param) for name, param in avg_model.named_parameters() if name in shared_layer}
    p_param = {name: copy.deepcopy(param) for name, param in avg_model.named_parameters() if name not in shared_layer}
    print([name for name in g_param])
    print(sum([torch.numel(g_param[name]) for name in g_param]) / (
            sum([torch.numel(g_param[name]) for name in g_param]) + sum(
        [torch.numel(p_param[name]) for name in p_param])))
    p_params = [copy.deepcopy(p_param) for i in range(n_client)]
    if not skip_training:
        for i in range(com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clients = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clients) != 0:
                    break

            # print('Communication Round', i + 1, flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            del client_models
            client_models = list(range(n_client))
            g_param_temp = copy.deepcopy(g_param)
            g_param = {name: 0 for name in g_param}
            for client in selected_clients:
                train_x = client_x[client]
                train_y = client_y[client]
                test_x = False
                test_y = False

                client_models[client] = model_func().to(device)
                w_full = {}
                for name in g_param_temp:
                    w_full[name] = g_param_temp[name]
                for name in p_param:
                    w_full[name] = p_params[client][name]
                client_models[client].load_state_dict(w_full)

                client_models[client] = train_model(client_models[client], train_x, train_y,
                                                    test_x, test_y,
                                                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch, 2,
                                                    weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma, device)
                for name in g_param:
                    g_param[name] += 1 / len(selected_clients) * copy.deepcopy(client_models[client].state_dict()[name])
                for name in p_params[0]:
                    p_params[client][name] = copy.deepcopy(client_models[client].state_dict()[name])

            if (i + 1) % test_per == 0:
                accs = []
                losses = []
                for idx in range(n_client):
                    temp_model = model_func()
                    w_full = {}
                    for name in g_param:
                        w_full[name] = g_param_temp[name]
                    for name in p_param:
                        w_full[name] = p_params[idx][name]
                    temp_model.load_state_dict(w_full)
                    loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                       temp_model, data_obj.dataset, device, 0)
                    accs.append(acc_test)
                    losses.append(loss_test)
                print("**** p  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
                test_acc1.append(accs)

        file_name = "{}_lgfedavg_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc1)
    return


def train_SCAFFOLD(data_obj, act_prob, learning_rate, batch_size, n_minibatch,
                   com_amount, test_per, weight_decay,
                   model_func, init_model, sch_step, sch_gamma, device, epoch, finetune_proximal=0.01,
                   lr_decay_per_round=1, rand_seed=0, global_learning_rate=1, iid="FALSE"):
    test_acc1 = []
    test_acc3 = []
    n_client = data_obj.n_client
    client_x = data_obj.client_x;
    client_y = data_obj.client_y

    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)

    weight_list = np.asarray([len(client_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client  # normalize it

    train_perf_sel = np.zeros((com_amount, 2));
    train_perf_all = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2));
    test_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_client + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    client_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_client X n_par
    client_models = list(range(n_client))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
    ditto_params = [copy.deepcopy(cld_mdl_param) for i in range(n_client)]

    for i in range(com_amount):
        inc_seed = 0
        while True:
            np.random.seed(i + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            # Choose at least one client in each synch
            if len(selected_clients) != 0:
                break

        print('Communication Round', i + 1, flush=True)
        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

        del client_models

        client_models = list(range(n_client))
        delta_c_sum = np.zeros(n_par)
        prev_params = get_mdl_params([avg_model], n_par)[0]

        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]

            client_models[client] = model_func().to(device)

            client_models[client].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

            for params in client_models[client].parameters():
                params.requires_grad = True

            # Scale down c
            state_params_diff_curr = torch.tensor(
                -state_params_diffs[client] + state_params_diffs[-1] / weight_list[client], dtype=torch.float32,
                device=device)

            client_models[client] = train_scaffold_mdl(client_models[client], model_func, state_params_diff_curr,
                                                       train_x, train_y,
                                                       learning_rate * (lr_decay_per_round ** i), batch_size,
                                                       n_minibatch, 1,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma, device,
                                                       print_verbose=True)

            curr_model_param = get_mdl_params([client_models[client]], n_par)[0]
            new_c = state_params_diffs[client] - state_params_diffs[-1] / weight_list[
                client] + 1 / n_minibatch / learning_rate / (lr_decay_per_round ** i) * (prev_params - curr_model_param)
            # Scale up delta c
            delta_c_sum += (new_c - state_params_diffs[client]) * weight_list[client]
            state_params_diffs[client] = new_c

            client_params_list[client] = curr_model_param

            org_weights = {}
            for name, param in avg_model.named_parameters():
                org_weights[name] = copy.deepcopy(param)
            client_models[client] = set_client_from_params(client_models[client].to(device), ditto_params[client],
                                                           device)
            # activate all layers
            for params in client_models[client].parameters():
                params.requires_grad = True
            client_models[client] = finetune_model(client_models[client],
                                                   train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                                   batch_size, 2, 1,
                                                   data_obj.dataset,
                                                   device,
                                                   finetune_proximal=finetune_proximal, org_weights=org_weights,
                                                   print_verbose=True)

            ditto_params[client] = get_mdl_params([client_models[client]], n_par)[0]

        avg_model_params = global_learning_rate * np.mean(client_params_list[selected_clients], axis=0) + (
                1 - global_learning_rate) * prev_params

        avg_model = set_client_from_params(model_func().to(device), avg_model_params, device)

        state_params_diffs[-1] += 1 / n_client * delta_c_sum

        all_model = set_client_from_params(model_func(), np.mean(client_params_list, axis=0), device)

        if (i + 1) % test_per == 0:
            accs = []
            losses = []
            accs2 = []
            losses2 = []
            for idx in range(n_client):
                loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                   avg_model, data_obj.dataset, device, 0)
                accs.append(acc_test)
                losses.append(loss_test)

                ditto_model = set_client_from_params(model_func(), ditto_params[idx], device)
                loss_test2, acc_test2 = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                     ditto_model, data_obj.dataset, device, 0)
                accs2.append(acc_test2)
                losses2.append(loss_test2)

            print("**** all  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
            print("**** ditto  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, np.mean(accs2), np.mean(losses2)), flush=True)
            test_acc1.append(accs)
            test_acc3.append(accs2)
        file_name = "{}_scaffold_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc1)

        file_name = "{}_ditto_scaffold_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
        record_npy(file_name, test_acc3)
    return


def count_rank_density(param):
    for name in param:
        # print(name)
        shape = param[name].shape
        # print(shape)
        comm_param_num = 0
        if len(shape) == 1:
            comm_param_num += torch.numel(param[name])
        else:
            if len(shape) == 2:
                matrix = param[name]
                U, S, VT = torch.linalg.svd(matrix, full_matrices=False)
                position = torch.nonzero(S, as_tuple=True)[0]
                U_comm = U[:, position] @ torch.sqrt(torch.diag(S[position]))
                VT_comm = torch.sqrt(torch.diag(S[position])) @ VT[position, :]
                comm_param_num += torch.numel(U_comm) + torch.numel(VT_comm)
            else:
                [n, m, d1, d2] = param[name].shape
                swapped = param[name].swapaxes(1, 2)
                matrix = swapped.reshape([n * d1, m * d2])
                # print(torch.norm(matrix))
                U, S, VT = torch.linalg.svd(matrix, full_matrices=False)
                position = torch.nonzero(S, as_tuple=True)[0]
                U_comm = U[:, position] @ torch.sqrt(torch.diag(S[position]))
                VT_comm = torch.sqrt(torch.diag(S[position])) @ VT[position, :]
                comm_param_num += torch.numel(U_comm) + torch.numel(VT_comm)

    return comm_param_num


def train_fedSLR(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, test_per,
                  weight_decay, model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, sparse_threshold, rank_threshold, device, skip_training=False, save_models=True,
                   rand_seed=0,
                  lr_decay_per_round=1, iid="FALSE", model_name="Resnet18", mode=None):

    def switch_training_mode(model, mode):
        for module in mask_modules(model):
            module.set_train_status(mode)

    def mask_modules(model):
        new_modules = []
        for module in model.modules():
            if isinstance(module, MaskedModule):
                new_modules.append(module)
        return new_modules

    def get_p_params(model):
        p_param = {}
        for name, param in model.named_parameters():
            if ".p" in name:
                p_param[name] = copy.deepcopy(param.data.to("cpu"))
        return p_param

    def get_w_params(model):
        w_param = {}
        for name, param in model.named_parameters():
            # print(name)
            if ".w" in name:
                # print(name)
                w_param[name] = copy.deepcopy(param.data.to("cpu"))
        return w_param

    def set_p_model_params(model, w, p, head=None):
        state_dict = {}
        for name in w:
            state_dict[name] = w[name].data
        for name in p:
            state_dict[name] = p[name].data
        # override the head if given
        if head != None:
            for name in head:
                state_dict[name] = head[name].data
        model.load_state_dict(state_dict, strict=False)
        return model

    def aggregate_w(local_w_aggregated, local_dual_variables_mu_server, w_threshold, device, mode):
        intact_w = {}
        for name in local_w_aggregated:
            avg_weights = 1 / (n_client)
            intact_w[name] = local_w_aggregated[name]
            for client_idx in range(n_client):
                intact_w[name] -= avg_weights * 1 / alpha_coef * local_dual_variables_mu_server[client_idx][name]
        cld_model_w, comm_param_num = SVD_st_update(intact_w, w_threshold, device)
        comm_param_num_array.append(comm_param_num)
        print("downlink_comm: {}".format(comm_param_num))
        return cld_model_w


    # def get_concat_w
    def SVD_st_update(param, w_threshold, device):
        comm_param_num = 0
        dense_rank = 0
        full_rank = 0
        with torch.no_grad():
            # print(param)
            # print(rank_density)
            new_param = {}
            for name in param:
                # print(name)
                param[name].to(device)
                shape = param[name].shape
                # print(shape)
                if len(shape) == 1:
                    new_param[name] = param[name]
                    comm_param_num += torch.numel(new_param[name])
                else:
                    if len(shape) == 2:
                        matrix = param[name]
                        U, S, VT = torch.linalg.svd(matrix, full_matrices=False)
                        reg = w_threshold / (alpha_coef)
                        S.data = (S.data - reg) * (S.data >= reg) + (S.data + reg) * (
                                S.data <= -reg)
                        position = torch.nonzero(S, as_tuple=True)[0]
                        dense_rank += len(position)
                        full_rank += torch.numel(S)
                        U_comm = U[:, position] @ torch.sqrt(torch.diag(S[position]))
                        VT_comm = torch.sqrt(torch.diag(S[position])) @ VT[position, :]
                        new_param[name] = U_comm @ VT_comm

                        if torch.numel(new_param[name]) > torch.numel(U_comm) + torch.numel(VT_comm):
                            comm_param_num += torch.numel(U_comm) + torch.numel(VT_comm)
                        else:
                            comm_param_num += torch.numel(new_param[name])

                    else:
                        [n, m, d1, d2] = param[name].shape
                        swapped = param[name].swapaxes(1, 2)
                        matrix = swapped.reshape([n * d1, m * d2])
                        # print(torch.norm(matrix))
                        U, S, VT = torch.linalg.svd(matrix, full_matrices=False)
                        reg = w_threshold / (alpha_coef)
                        # print(reg)
                        S.data = (S.data - reg) * (S.data >= reg) + (S.data + reg) * (
                                S.data <= -reg)

                        position = torch.nonzero(S, as_tuple=True)[0]
                        dense_rank += len(position)
                        full_rank += torch.numel(S)
                        U_comm = U[:, position] @ torch.sqrt(torch.diag(S[position]))
                        VT_comm = torch.sqrt(torch.diag(S[position])) @ VT[position, :]
                        new_param[name] = (U_comm @ VT_comm).reshape([n, d1, m, d2]).swapaxes(1, 2)
                        # new_param[name] = param[name]
                        if torch.numel(new_param[name]) > torch.numel(U_comm) + torch.numel(VT_comm):
                            comm_param_num += torch.numel(U_comm) + torch.numel(VT_comm)
                        else:
                            comm_param_num += torch.numel(new_param[name])
        print(dense_rank / full_rank)
        return new_param, comm_param_num

    def concat_param(param):
        concat_param = {}
        for name in param:
            if not isinstance(concat_param, torch.Tensor):
                # Initially nothing to concatenate
                concat_param = param[name].reshape(-1)
            else:
                concat_param = torch.cat((concat_param, param[name].reshape(-1)), 0)
        return concat_param

    n_client = data_obj.n_client
    client_x = data_obj.client_x;
    client_y = data_obj.client_y

    test_acc1 = []
    test_acc3 = []
    comm_param_num_array = []
    sparse_param_num_array = []
    cld_model_w = get_w_params(init_model)
    print([name for name in cld_model_w])
    cld_model_p = get_p_params(init_model)
    dual_variable_mu = {name: torch.zeros_like(cld_model_w[name]) for name in cld_model_w}
    local_p = [copy.deepcopy(cld_model_p) for i in range(n_client)]
    local_dual_variables_mu = [copy.deepcopy(dual_variable_mu) for i in range(n_client)]
    previous_mu = [[] for i in range(n_client)]
    del cld_model_p
    del dual_variable_mu
    client_models = list(range(n_client))
    if not skip_training:
        for i in range(com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clients = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clients) != 0:
                    break

            print('Communication Round', i + 1, flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))
            # cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            local_w_aggregated = {name: 0 for name in cld_model_w}
            concat_w = concat_param(cld_model_w)

            for client in selected_clients:
                print("client number {}".format(client))
                train_x = client_x[client]
                train_y = client_y[client]

                model = model_func()

                alpha_coef_adpt = alpha_coef  # adaptive alpha coef

                concat_mu = concat_param(local_dual_variables_mu[client])
                # if len(previous_mu[client])>0:
                #     concat_mu += gamma*(concat_mu - previous_mu[client] )
                model = set_p_model_params(model, cld_model_w, local_p[client])
                model = train_model_fedSLR(model, model_func, alpha_coef_adpt,
                                            concat_w, concat_mu,
                                            train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                            batch_size, epoch, 1, weight_decay,
                                            data_obj.dataset, sch_step, sch_gamma, sparse_threshold,
                                            device, client_num=n_client, print_verbose=True, mode=mode)

                curr_w_par = get_w_params(model)
                for name in local_w_aggregated:
                    local_w_aggregated[name] += 1 / len(selected_clients) * copy.deepcopy(curr_w_par[name])
                previous_mu[client] = copy.deepcopy(concat_mu)

                # cancel the error resulted from factorization
                for name in curr_w_par:
                    local_dual_variables_mu[client][name] += alpha_coef * (cld_model_w[name] - curr_w_par[name])

                # finetune the sparse component
                model = set_p_model_params(model, cld_model_w, local_p[client])
                switch_training_mode(model, "train_p")
                model = finetune_model(model, train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                       batch_size, 1, 1,
                                       data_obj.dataset,
                                       device, se_threshold=sparse_threshold, org_weights={}, print_verbose=True,
                                       mode=mode)

                curr_p_par = get_p_params(model)
                local_p[client] = copy.deepcopy(curr_p_par)

            temp = copy.deepcopy(cld_model_w)
            cld_model_w = aggregate_w(local_w_aggregated, local_dual_variables_mu, rank_threshold, device, mode)

            if (i + 1) % test_per == 0:
                accs = []
                losses = []
                accs3 = []
                losses3 = []
                non_zero = []
                cur_cld_model = model_func().to(device)

                for idx in range(n_client):
                    cur_cld_model = set_p_model_params(cur_cld_model, cld_model_w, {})
                    switch_training_mode(cur_cld_model, "only_w")
                    loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                       cur_cld_model, data_obj.dataset, device, None)
                    accs.append(acc_test)
                    losses.append(loss_test)

                    switch_training_mode(cur_cld_model, "train_p")
                    cur_cld_model = set_p_model_params(cur_cld_model, cld_model_w, local_p[idx], {})

                    loss_test3, acc_test3 = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                         cur_cld_model, data_obj.dataset, device, None)
                    accs3.append(acc_test3)
                    losses3.append(loss_test3)
                    non_zero.append(0)
                    for name in local_p[idx]:
                        non_zero[idx] += torch.count_nonzero(local_p[idx][name])

                sparse_param_num_array.append(non_zero)

                del cur_cld_model
                print("**** Cur  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
                test_acc1.append(accs)
                print("**** Cur low rank + sparse model %3d, Test Accuracy: %.4f, Loss: %.4f"
                      % (i + 1, np.mean(accs3), np.mean(losses3)), flush=True)
                test_acc3.append(accs3)

        file_name = "{}_g_fedlite_sr{}_rank{}_epoch{}_alpha{}_split{}_model{}_mode{}".format(data_obj.dataset,
                                                                                             sparse_threshold,
                                                                                             rank_threshold, epoch,
                                                                                             alpha_coef, iid,
                                                                                             model_name, mode)
        record_npy(file_name, test_acc1)

        file_name = "{}_p_fedlite_sr{}_rank{}_epoch{}_alpha{}_split{}_model{}_mode{}".format(data_obj.dataset,
                                                                                             sparse_threshold,
                                                                                             rank_threshold, epoch,
                                                                                             alpha_coef, iid,
                                                                                             model_name, mode)
        record_npy(file_name, test_acc3)

        file_name = "{}_comm_sr{}_rank{}_epoch{}_alpha{}_split{}_model{}_mode{}".format(data_obj.dataset,
                                                                                        sparse_threshold,
                                                                                        rank_threshold, epoch,
                                                                                        alpha_coef, iid, model_name,
                                                                                        mode)
        record_npy(file_name, comm_param_num_array)

        file_name = "{}_sparse_sr{}_rank{}_epoch{}_alpha{}_split{}_model{}_mode{}".format(data_obj.dataset,
                                                                                          sparse_threshold,
                                                                                          rank_threshold, epoch,
                                                                                          alpha_coef, iid, model_name,
                                                                                          mode)
        record_npy(file_name, sparse_param_num_array)

        if save_models:
            cur_cld_model = model_func()
            for idx in range(n_client):
                cur_cld_model = set_p_model_params(cur_cld_model, cld_model_w, local_p[idx])
                Path = "model/{}_client_model_".format(data_obj.dataset) + str(idx) + "_" + str(
                    rank_threshold) + "_" + str(sparse_threshold) + "_" + str(model_name) + "_" + str(mode)
                torch.save({
                    'model_state_dict': cur_cld_model.state_dict(),
                }, Path)
            del cur_cld_model


def train_apfl(data_obj, act_prob,
               learning_rate, batch_size, epoch, com_amount, test_per,
               weight_decay, model_func, init_model,
               sch_step, sch_gamma, device, alpha_apfl=0.5, rand_seed=0,
               lr_decay_per_round=1, iid="FALSE"):
    n_client = data_obj.n_client

    client_x = data_obj.client_x;
    client_y = data_obj.client_y
    test_acc1 = []
    test_acc3 = []
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    client_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_client X n_par

    client_models = list(range(n_client))
    avg_model = model_func()
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
    vs = [copy.deepcopy(dict(avg_model.named_parameters())) for client in range(n_client)]
    for i in range(com_amount):
        inc_seed = 0
        while (True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                break

        # print('Communication Round', i + 1, flush=True)
        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

        del client_models
        client_models = list(range(n_client))
        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]
            test_x = False
            test_y = False

            client_models[client] = model_func().to(device)
            client_models[client].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
            comm_param_num = 0
            for name, param in avg_model.named_parameters():
                comm_param_num += torch.numel(param)

            client_models[client], vs[client] = train_model_apfl(client_models[client], train_x, train_y,
                                                                 test_x, test_y,
                                                                 learning_rate * (lr_decay_per_round ** i), batch_size,
                                                                 epoch, 1,
                                                                 weight_decay,
                                                                 data_obj.dataset, sch_step, sch_gamma, device,
                                                                 v=vs[client], alpha_apfl=alpha_apfl)

            client_params_list[client] = get_mdl_params([client_models[client]], n_par)[0]

        avg_model = set_client_from_params(model_func(), np.mean(client_params_list[selected_clients], axis=0),
                                           device)
        # all_model = set_client_from_params(model_func(), np.mean(client_params_list, axis=0))

        if (i + 1) % test_per == 0:

            accs = []
            losses = []
            for idx in range(n_client):
                temp_dict = {}
                for name in vs[0]:
                    temp_dict[name] = (1 - alpha_apfl) * avg_model.state_dict()[name].to("cpu") + alpha_apfl * vs[idx][
                        name].to("cpu")
                temp_model = model_func()
                temp_model.load_state_dict(temp_dict)
                loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                   temp_model, data_obj.dataset, device, 0)
                accs.append(acc_test)
                losses.append(loss_test)

            print("**** p  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
            test_acc1.append(accs)

    file_name = "{}_APFL_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
    record_npy(file_name, test_acc1)
    return


def train_standalone(data_obj, act_prob,
                     learning_rate, batch_size, epoch, com_amount, test_per,
                     weight_decay, model_func, init_model,
                     sch_step, sch_gamma, device, rand_seed=0,
                     lr_decay_per_round=1, iid="FALSE"):
    n_client = data_obj.n_client

    client_x = data_obj.client_x;
    client_y = data_obj.client_y
    test_acc1 = []
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    client_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_client X n_par

    client_models = list(range(n_client))
    avg_model = model_func()
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
    local_ws = [copy.deepcopy(dict(avg_model.named_parameters())) for client in range(n_client)]
    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    for i in range(com_amount):
        inc_seed = 0
        while (True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                break

        # print('Communication Round', i + 1, flush=True)
        # print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

        del client_models
        client_models = list(range(n_client))
        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]
            test_x = False
            test_y = False

            client_models[client] = model_func().to(device)
            client_models[client].load_state_dict(local_ws[client])
            comm_param_num = 0
            for name, param in avg_model.named_parameters():
                comm_param_num += torch.numel(param)

            client_models[client] = train_model(client_models[client], train_x, train_y,
                                                test_x, test_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, 1,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma, device)

            local_ws[client] = copy.deepcopy(client_models[client].to("cpu").state_dict())

        if (i + 1) % test_per == 0:
            accs = []
            losses = []
            for idx in range(n_client):
                temp_model = model_func()
                temp_model.load_state_dict(local_ws[idx])
                loss_test, acc_test = get_acc_loss(data_obj.client_test_x[idx], data_obj.client_test_y[idx],
                                                   temp_model, data_obj.dataset, device, 0)
                accs.append(acc_test)
                losses.append(loss_test)

            print("**** p  w model  %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, np.mean(accs), np.mean(losses)), flush=True)
            test_acc1.append(accs)
            file_name = "{}_standalone_epoch{}_split{}".format(data_obj.dataset, epoch, iid)
            record_npy(file_name, test_acc1)
    return
