import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Dropout2d, ReLU, Flatten, Linear, BatchNorm2d, AvgPool2d, Sigmoid, Tanh, Softmax, Dropout, ConvTranspose2d, BatchNorm1d, ModuleDict

from util import one_hot_image
######### MLP #########

class Toy_MLP(Module):
    def __init__(self, input_dim, layers, hidden_num, p_drop = 0.2):
        super().__init__()
        self.NN = Sequential(Linear(input_dim + 1, hidden_num, bias=True))

        self.p_drop = p_drop

        for layer in range(layers -2):
            self.NN.append(Tanh())
            self.NN.append(Dropout(p_drop))
            self.NN.append(Linear(hidden_num, hidden_num, bias=True))
        self.NN.append(Tanh())
        self.NN.append(Dropout(p_drop))
        self.NN.append(Linear(hidden_num, input_dim, bias=True))
                            
    
    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.NN(inputs)

        return x

class Toy_MLP_distill(Module):
    def __init__(self, input_dim, layers, hidden_num, p_drop = 0.2):
        super().__init__()
        self.NN = Sequential(Linear(input_dim, hidden_num, bias=True))

        self.p_drop = p_drop

        for layer in range(layers -2):
            self.NN.append(Tanh())
            self.NN.append(Dropout(p_drop))
            self.NN.append(Linear(hidden_num, hidden_num, bias=True))
        self.NN.append(Tanh())
        self.NN.append(Dropout(p_drop))
        self.NN.append(Linear(hidden_num, input_dim, bias=True))
                            
    
    def forward(self, x_input):
        x = self.NN(x_input)

        return x

class MLP(Module):
    def __init__(self, input_dim, layers, hidden_num, p_drop = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.input_dim = hidden_num

        self.p_drop = p_drop

        self.NN = Sequential(Linear(input_dim + 1, hidden_num, bias=True),
                            ReLU(),
                            BatchNorm1d(hidden_num)
                            )

        for layer in range(layers -2):
            self.NN.append(Dropout(p_drop))
            self.NN.append(Linear(hidden_num, hidden_num, bias=True))
            self.NN.append(ReLU())
            self.NN.append(BatchNorm1d(hidden_num))
        self.NN.append(Dropout(p_drop))
        self.NN.append(Linear(hidden_num, input_dim, bias=True))


    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.NN(inputs)

        return x

############## U-Net ##############

class U_Net(Module):
    def contracting_block(self, size_in, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out),
                           Conv2d(size_out, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def expanding_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           ConvTranspose2d(size_mid, size_out, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
                           )
        return block

    def finalizing_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_out, kernel_size=1),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def __init__(self, input_shape, output_channels):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.output_channels = output_channels

        self.batches = []
        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        self.NN = ModuleDict({
            "contr_1": self.contracting_block(input_shape[1], 64, 3),
            "contr_2": self.contracting_block(64, 128, 3),
            "contr_3": self.contracting_block(128, 256, 3),
            "bottleneck": self.expanding_block(256, 512, 256, 3),
            "expand_1": self.expanding_block(512, 256, 128, 3),
            "expand_2": self.expanding_block(256, 128, 64, 3),
            "final": self.finalizing_block(128, 64, output_channels, 3),
            "max_pool": MaxPool2d(2)
        })

    def forward(self, x):
        skip_1 = self.NN['contr_1'](x)
        contr_1 = self.NN["max_pool"](skip_1)
        skip_2 = self.NN["contr_2"](contr_1)
        contr_2 = self.NN["max_pool"](skip_2)
        skip_3 = self.NN["contr_3"](contr_2)
        contr_3 = self.NN["max_pool"](skip_3)

        bottleneck = self.NN["bottleneck"](contr_3)

        cat_1 = torch.cat((bottleneck, skip_3), 1)
        exp_1 = self.NN["expand_1"](cat_1)
        cat_2 = torch.cat((exp_1, skip_2), 1)
        exp_2 = self.NN["expand_2"](cat_2)
        cat_3 = torch.cat((exp_2, skip_1), 1)
        final = self.NN["final"](cat_3)

        return final


class U_Net_big(Module):
    def contracting_block(self, size_in, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out),
                           Conv2d(size_out, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def expanding_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           ConvTranspose2d(size_mid, size_out, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
                           )
        return block

    def finalizing_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_out, kernel_size=1),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def __init__(self, input_shape, output_channels, device):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.output_channels = output_channels
        self.device = device

        self.batches = []
        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        self.NN = ModuleDict({
            "contr_1": self.contracting_block(input_shape[1], 64, 3),
            "contr_2": self.contracting_block(64, 128, 3),
            "contr_3": self.contracting_block(128, 256, 3),
            "contr_4": self.contracting_block(256, 512, 3),
            "bottleneck": self.expanding_block(512, 1024, 512, 3),
            "expand_1": self.expanding_block(1024, 512, 256, 3),
            "expand_2": self.expanding_block(512, 256, 128, 3),
            "expand_3": self.expanding_block(256, 128, 64, 3),
            "final": self.finalizing_block(128, 64, output_channels, 3),
            "max_pool": MaxPool2d(2)
        })

    def forward(self, t, x, c=None):
        t_expand = t.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        if c != None:
            c_expand = one_hot_image(c, self.device)
            x_input = torch.cat([x, t_expand, c_expand], axis=1).reshape(-1, x.shape[1] + 2,32,32)
        else:
            x_input = torch.cat([x, t_expand], axis=1).reshape(-1, x.shape[1] + 1,32,32)

        skip_1 = self.NN['contr_1'](x_input)
        contr_1 = self.NN["max_pool"](skip_1)
        skip_2 = self.NN["contr_2"](contr_1)
        contr_2 = self.NN["max_pool"](skip_2)
        skip_3 = self.NN["contr_3"](contr_2)
        contr_3 = self.NN["max_pool"](skip_3)
        skip_4 = self.NN["contr_4"](contr_3)
        contr_4 = self.NN["max_pool"](skip_4)

        bottleneck = self.NN["bottleneck"](contr_4)

        cat_1 = torch.cat((bottleneck, skip_4), 1)
        exp_1 = self.NN["expand_1"](cat_1)
        cat_2 = torch.cat((exp_1, skip_3), 1)
        exp_2 = self.NN["expand_2"](cat_2)
        cat_3 = torch.cat((exp_2, skip_2), 1)
        exp_3 = self.NN["expand_3"](cat_3)
        cat_4 = torch.cat((exp_3, skip_1), 1)
        final = self.NN["final"](cat_4)

        return final
    

class U_Net_big_cond(Module):
    def contracting_block(self, size_in, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out),
                           Conv2d(size_out, size_out, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def expanding_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           ConvTranspose2d(size_mid, size_out, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
                           )
        return block

    def finalizing_block(self, size_in, size_mid, size_out, kernel):
        block = Sequential(Conv2d(size_in, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_mid, kernel, padding="same"),
                           ReLU(),
                           BatchNorm2d(size_mid),
                           Conv2d(size_mid, size_out, kernel_size=1),
                           ReLU(),
                           BatchNorm2d(size_out)
                           )
        return block

    def __init__(self, input_shape, output_channels, device):
        '''input_shape: tuple (batch_size, channels, x_pixels, y_pixels)'''
        super().__init__()

        self.input_shape = input_shape
        self.output_channels = output_channels
        self.device = device

        self.batches = []
        self.train_loss = []
        self.train_time = []
        self.test_loss = []
        self.test_accuracy = []

        self.NN = ModuleDict({
            "contr_1": self.contracting_block(input_shape[1], 64, 3),
            "contr_2": self.contracting_block(64, 128, 3),
            "contr_3": self.contracting_block(128, 256, 3),
            "contr_4": self.contracting_block(256, 512, 3),
            "bottleneck": self.expanding_block(512, 1024, 512, 3),
            "expand_1": self.expanding_block(1024, 512, 256, 3),
            "expand_2": self.expanding_block(512, 256, 128, 3),
            "expand_3": self.expanding_block(256, 128, 64, 3),
            "final": self.finalizing_block(128, 64, output_channels, 3),
            "max_pool": MaxPool2d(2)
        })

    def forward(self, t, x, c=None):
        t_expand = t.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        # if c != None:
        c_expand = torch.zeros((c.shape[0], 1, 32, 32), device=self.device)
        c_expand[:, 0, 0, :16] = c
        x_input = torch.cat([x, t_expand, c_expand], axis=1).reshape(-1, x.shape[1] + 2,32,32)
        # else:
            # x_input = torch.cat([x, t_expand], axis=1).reshape(-1, x.shape[1] + 1,32,32)

        skip_1 = self.NN['contr_1'](x_input)
        contr_1 = self.NN["max_pool"](skip_1)
        skip_2 = self.NN["contr_2"](contr_1)
        contr_2 = self.NN["max_pool"](skip_2)
        skip_3 = self.NN["contr_3"](contr_2)
        contr_3 = self.NN["max_pool"](skip_3)
        skip_4 = self.NN["contr_4"](contr_3)
        contr_4 = self.NN["max_pool"](skip_4)

        bottleneck = self.NN["bottleneck"](contr_4)

        cat_1 = torch.cat((bottleneck, skip_4), 1)
        exp_1 = self.NN["expand_1"](cat_1)
        cat_2 = torch.cat((exp_1, skip_3), 1)
        exp_2 = self.NN["expand_2"](cat_2)
        cat_3 = torch.cat((exp_2, skip_2), 1)
        exp_3 = self.NN["expand_3"](cat_3)
        cat_4 = torch.cat((exp_3, skip_1), 1)
        final = self.NN["final"](cat_4)

        return final