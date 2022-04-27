import torch.nn as nn
import torch
from torch import batch_norm, conv1d, nn as nn
import torch.nn.functional as F


def square_distance(centroid, pointcloud):
    """
    Calculate the Euclidean distance between point-wise.

    2 * centroid * pointcloud^T = 2 * xp * xn + 2 * yp * yn + 2 * zp * zn; inner product!
    sum(centroid^2, dim=-1) = xp * xp + yp * yp + zp * zp;
    sum(pointcloud^2, dim=-1) = xn * xn + yn * yn + zn * zn;

    Input:
        centroid: centroids, [B, P, 3]
        pointcloud: point clouds, [B, N, 3]
    Output:
        dist: per-point square distance, [B, P, N]
    """
    B, N, _ = pointcloud.shape
    _, P, _ = centroid.shape
    dist = -2 * torch.matmul(centroid, pointcloud.transpose(2, 1)) # [B,P,N]
    dist += torch.sum(pointcloud ** 2, -1).view(B, 1, N) # [B,1,N]
    dist += torch.sum(centroid ** 2, -1).view(B, P, 1)  # [B,1,N]
    return dist

def rps(pointcloud,p):
    """
    Randomly generate p centroids.
    """
    # [B,n,3]
    batch_size,N,_=pointcloud.shape
    idx = torch.randint(low=0,high=N,size=(batch_size,p),dtype=torch.long).to(pointcloud.device).view(batch_size,p,1).repeat(1,1,3)
    centroid = torch.gather(pointcloud,1,idx)
    # [B,p,3]
    return centroid

def knn(point,feat,centroid,p,K,D):
    """
    Point: [B,N,3]
    Feat: [B,N,f]
    centroid: [B,p,3]
    p: number of centroids
    K: size of kernel
    D: dilation rate
    return:
        Point: [B,p,K,3]
        Feat: [B,p,K,f]
    """
    batch_size,_,_ = point.shape
    dist = square_distance(centroid=centroid,pointcloud=point)
    # dist: [B,P,N]
    _,g = torch.topk(dist,K*D,dim=-1,largest=False,sorted=False)
    # g: [B,P,K*D]
    idx = torch.randperm(K*D,dtype=torch.long).to(point.device)[:K].view(1,1,K).repeat(batch_size,p,1)
    g = torch.gather(g,-1,idx).view(batch_size,p,K,1)
    # g: [B,P,K]
    point = point.view(batch_size,1,-1,3).repeat(1,p,1,1)
    # Point: [B, P, N, 3]
    point = torch.gather(point,2,g.repeat(1,1,1,3))
    centroid = centroid.view(batch_size,p,1,3).repeat(1,1,K,1)
    point = point - centroid
    # Feat
    if feat != None:
        feat = feat.reshape(batch_size,1,-1,feat.shape[-1]).repeat(1,p,1,1)
        feat = torch.gather(feat,2,g.repeat(1,1,1,feat.shape[-1]))
    return point,feat


class Xconv(nn.Module):
    def __init__(self,in_channel,out_channel,P,K,D):
        """
        By default lifted channel C_delta = C1/4
        in_channel: C1
        out_channel: C2
        p: number of centroids
        k: kernel size
        D: dilation rate, normally 2
        """
        super().__init__()
        self.c1 = in_channel
        self.c2 = out_channel
        if P == -1:
            self.p =0
        else:
            self.p = P
        self.k = K
        self.d = D
        if in_channel == 0:
            lift_channel = 4
        else:
            lift_channel = int(in_channel/4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=lift_channel,kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel),
            nn.Conv2d(in_channels=lift_channel,out_channels=lift_channel,kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel)
        )
        self.conv2 = nn.Conv1d(in_channels=self.k*3,out_channels=self.k*self.k,kernel_size=1)
        self.bn2 = nn.BatchNorm1d(self.k*self.k)
        self.conv3 = nn.Conv1d(in_channels=self.k,out_channels=self.k,kernel_size=1)
        self.bn3 = nn.BatchNorm1d(self.k)
        self.conv31 = nn.Conv1d(in_channels=self.k,out_channels=self.k,kernel_size=1)
        self.bn31 = nn.BatchNorm1d(self.k)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.k,out_channels=self.k,kernel_size=1),
            nn.ELU(),
            nn.BatchNorm2d(self.k),
            nn.Conv2d(in_channels=self.k,out_channels=3,kernel_size=1),
            nn.ELU(),
            nn.BatchNorm2d(3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d((lift_channel+in_channel)*3,out_channels=out_channel,kernel_size=1),
            nn.ELU(),
            nn.BatchNorm1d(out_channel)
        )
        self.elu = nn.ELU(inplace=True)

    def forward(self,P,Feat):
        """
        P: [B,N,3]
        Feat: [B,N,f]
        """
        batch_size = P.shape[0]
        if self.p == 0:
            self.p = P.shape[1]
        centroid = rps(P,self.p)
        Point,Feat = knn(point=P,feat=Feat,centroid=centroid,p=self.p,K=self.k,D=self.d)
        # P: [B,P,K,3]
        # Feat: [B,P,K,f]
        Point = Point.permute(0,3,1,2)
        # P: [B,3,P,K]
        if Feat == None:
            Feat = self.conv1(Point).permute(0,2,3,1)
        else:
            F_delta = self.conv1(Point).permute(0,2,3,1)
            Feat = torch.cat([Feat,F_delta],dim=-1)
            # Step 1 concatenation finished
        
        # Feat: [B,P,K,f]
        # P: [B,3,P,K]
        Point = Point.permute(0,2,3,1)
        Point = Point.reshape(Point.shape[0],Point.shape[1],-1).transpose(2,1)
        # P: [B,3K,P]
        X = self.bn2(self.elu(self.conv2(Point)))
        X = X.transpose(2,1).reshape(-1,self.p,self.k,self.k).permute(0,3,1,2).reshape(-1,self.k,self.p*self.k)
        # X: [B,P,K,K]
        X = self.bn3(self.elu(self.conv3(X)))
        X = self.bn31(self.elu(self.conv31(X)))
        # X: [B,K,P*K]
        X = X.reshape(-1,self.k,self.p,self.k).permute(0,2,3,1)
        # X: [B,P,K,K]
        # Step 2 X generation

        Feat = torch.matmul(X,Feat)
        # Feat: [B,p,k,f]
        Feat = Feat.transpose(2,1) # Feat: [B,k,p,f]
        Feat = self.conv4(Feat).transpose(2,1).reshape(batch_size,self.p,-1).transpose(2,1) # Feat: [B,2*f,p]
        Feat = self.conv5(Feat).transpose(2,1)
        # Feat: [B,p,c2]
        # Step 3 Conv(K,F)
        return centroid, Feat


class PointCNNModule(nn.Module):
    """
    do not take in extra features in this version.
    """
    def __init__(self,feature,num_class):
        super().__init__()
        self.f = feature
        self.xc1 = Xconv(feature,96,-1,16,2)
        self.xc2 = Xconv(96,96,768,16,2)
        self.xc3 = Xconv(96,192,384,16,2)
        self.xc4 = Xconv(192,384,128,16,6)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=384,out_channels=96,kernel_size=1),
            nn.ELU(),
            nn.BatchNorm1d(96),
            nn.Conv1d(in_channels=96,out_channels=num_class,kernel_size=1),
            nn.ELU(),
            nn.BatchNorm1d(num_class),
            nn.Dropout()
        )
        self.elu = nn.ELU()
        
    def forward(self,**inputs):
        x = inputs['points']
        if self.f == 0:
            Feat = None
        else:
            Feat = inputs['features']
        # x.shape = [B,N,3]
        P,Feat = self.xc1(x,Feat)
        P,Feat = self.xc2(P,Feat)
        P,Feat = self.xc3(P,Feat)
        _,Feat = self.xc4(P,Feat)
        batch_size = Feat.shape[0]
        Feat = Feat.transpose(2,1)
        Feat = self.fc(Feat)
        Feat = torch.mean(Feat,dim=-1)
        # Feat.shape = [B,num_class]
        return Feat


class PointCNN(nn.Module):
    def __init__(self,feature,num_class):
        """
        (extra) features = 0
        num_class set to the number of class
        """
        super().__init__()
        self.encoder = PointCNNModule(feature,num_class=num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self,**input):
        features = self.encoder(**input)
        label_logit = features
        return {'pred_label_logit': label_logit}
    
