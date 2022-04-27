import torch
from torch import nn as nn
import torch.nn.functional as F

def knn(x, k):  # x.shape = (1, 3, 2048) = (batch_size, features, num_points)
    inner = -2*torch.matmul(x.transpose(2, 1), x)   # inner.shape = (1, 2048, 2048)
    xx = torch.sum(x**2, dim=1, keepdim=True)   # xx.shape = (1, 1, 2048)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)    # pairwise_distance.shape = (1, 3, 3)
    # print("the k is", k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)  # x.size(0) = x.shape[0]    # x.shape = (batch_size, num_points, features) = (1, 2048, 3)
    num_points = x.size(2)
    """
        tensor.view() Returns a new tensor with the same data
    """
    x = x.view(batch_size, -1, num_points)  #the size -1 is inferred from other dimensions (1, 3, 2048)
    if idx is None:
        idx = knn(x, k=k)   # idx.shape = (batch_size, num_points, k) (1, 2048, 20)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size() # (1, 3, 2048)

    x = x.transpose(2, 1).contiguous()   # (1, 2048, 3)# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) # (1, 2048, 20, 3)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() #(1, 6, 2048, 20)
  
    return feature

class DGCNNModule(nn.Module):
    """
    in_channels (int) - Number of channels in the input image
    out_channels (int) - Number of channels produced by the convolution, that is the number of filters
    kernel_size - Size of the convolving kernel
    bias -  If True, adds a learnable bias to the output. Default: True
    """
    def __init__(
                self,
                in_channel,
                num_class,
                emb_dims,
                out_dropout,
                k
                ):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)    # dimension of embeddings

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel*2, 64, kernel_size=1, bias=False), # 因为features和原来的x连接了，这里的in_channels 应为原来的2倍
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=out_dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=out_dropout)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, **inputs):
        x = inputs['points']    # x.shape = (1, 2048, 3)
        batch_size = x.size(0)
        num_points = x.size(1)
        x =  x.view(batch_size, -1, num_points)
        x = get_graph_feature(x, k=self.k)  # (batch_size, cat_features, num_points, k)
        x = self.conv1(x)   # x.shape = (1, 64, 2048, 20)   (batch_size, features, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    #(1, 64, 2048)

        x = get_graph_feature(x1, k=self.k) # (1, 128, 2048, 20)
        x = self.conv2(x)   #(1, 64, 2048, 20)
        x2 = x.max(dim=-1, keepdim=False)[0]    #(1, 64, 2048)

        x = get_graph_feature(x2, k=self.k) #(1, 128, 2048, 20)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]    #(1, 128, 2048)

        x = get_graph_feature(x3, k=self.k) #(1, 256, 2048, 20)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]    #(1, 256, 2048)

        x = torch.cat((x1, x2, x3, x4), dim=1)  #(1, 512, 2048)

        x = self.conv5(x)   #(1, 1024, 2048)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   #(1, 1024)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)   #(1, 1024)
        x = torch.cat((x1, x2), 1)  #(1, 2048)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x