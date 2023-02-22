
from resnet import customized_resnet18, customized_resnet18_k11
from utils_libs import *
import torchvision.models as models
from mask_component_bi import MaskedConv, MaskedLinear, MaskedGroupNorm

def construct_mask_model( model, args):
    model = copy.deepcopy(model)

    def do_mask(model, name):
        childrens = list(model.named_children())
        if not childrens:
            if isinstance(model, torch.nn.Conv2d):
                new_model = MaskedConv(model.weight, name, model.stride,
                                       model.padding, model.dilation, model.groups, args,
                                       device=model.weight.device,)
            elif isinstance(model, torch.nn.Linear):
                new_model = MaskedLinear(model.weight, name, args,
                                             device=model.weight.device, bias=model.bias)

            elif isinstance(model, torch.nn.GroupNorm):
                if args.method == "FedPara":
                    new_model=model
                else:
                    new_model = MaskedGroupNorm(model.num_groups, model.num_channels, model.weight, model.bias, name,
                                            args,
                                            device=model.weight.device)
            else:
                new_model = model
            return new_model

        for child_name, c in childrens:
            full_name = name + "." + child_name if name != "" else child_name
            new_model = do_mask(c, full_name)
            model.add_module(child_name, new_model)
        return model

    model = do_mask(model, "")
    return model

class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        self.args=args
        if self.name == 'Linear':
            # [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(3 * 32 * 32, 100)
          
        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'Resnet18':
            self.model = customized_resnet18()


        if self.name == "tinyimagenet_Resnet18":
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 200)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18
            # self.model = customized_resnet18(class_num=200)

        if self.name == 'Cifar100_Resnet18':

            self.model = customized_resnet18(class_num=100)

        if self.name == 'Densenet121':
            self.model = models.resnet18()
            self.model.fc = nn.Linear(512, 10)

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80
            
            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)

        
    def forward(self, x):
        if self.name == 'Linear':
            x = x.view(-1, 3 * 32 * 32)
            x = self.fc(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'Resnet18' or self.name == 'Cifar100_Resnet18' or self.name == "tinyimagenet_Resnet18" or self.name =="CIFAR100_k11resnet18" or self.name =="Cifar100_alexnet":
            x = self.model(x)

        if self.name == 'Densenet121':
            x = self.model(x)

        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2) # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1,:,:]
            x = self.fc(last_hidden)

        return x


