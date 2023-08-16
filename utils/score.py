from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import sklearn.covariance
from torch.autograd import Variable
import sklearn.metrics as sk

recall_level_default = 0.95

def get_measures(pos, neg):
    id_labels = np.ones_like(pos)
    ood_labels = np.zeros_like(neg)
    examples = np.concatenate((pos, neg))
    labels = np.concatenate((id_labels, ood_labels))

    aupr = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)
    fprs, tprs, _ = sk.roc_curve(labels, examples)
    fpr = fprs[np.where(tprs >= 0.95)[0][0]]

    return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, inf, method_name='Ours', recall_level=recall_level_default):
    print(method_name)
    print('\tFPR{:d} | AUROC | AU-PR | TIME'.format(int(100*recall_level)))
    print('\t{:5.2f} | {:5.2f} | {:5.2f} | {:.2f}\n'.format(100*fpr, 100*auroc, 100*aupr, inf))


def get_odin_score(inputs, outputs, model, temper, noiseMagnitude1):
    criterion = nn.CrossEntropyLoss()
    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    # Using temperature scaling
    outputs = outputs / temper
    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()
    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs, _ = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs


def get_gem_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, num_batches, in_dist=False,GEM=1):
    '''
    If GEM!=1 then computes the proposed Mahalanobis confidence score on input dataset
    If GEM=1 then computes the GEM score on input dataset
    return: GEM or Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    M_dist=[]
    
    for batch_idx, (data, target) in enumerate(test_loader):
        #print("this is target: ", target)
        if batch_idx >= num_batches and in_dist is False:
            break
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        #tempInputs = torch.add(data.data, -magnitude, gradient) #The useage is depreciated
        tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
        with torch.no_grad():
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)
        #print(noise_gaussian_score.data.cpu().numpy().shape)
        my_target=target.data.cpu().numpy()
        my_target=my_target.reshape(my_target.shape[0],-1)
        my_score=noise_gaussian_score.data.cpu().numpy()
        #print(my_target.shape,my_score.shape)
        my_data=np.concatenate((my_score,my_target),axis=1)
        M_dist.append(my_data)

        #PEYMAN
        if (GEM==1):
            noise_gaussian_score=torch.logsumexp(noise_gaussian_score, dim=1)
        else:
            noise_gaussian_score,_=torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32) #, np.concatenate(M_dist,axis=0)


def get_mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, num_batches, in_dist=False):
    model.eval()
    Mahalanobis = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32)


def sample_estimator(model, num_classes, feature_list, train_loader):
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision