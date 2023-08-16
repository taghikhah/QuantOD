import os
import ast
import numpy as np
import pandas as pd
import time as tm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models.mobilenet import MobileNetV2
from models.resnet import ResNet18
from models.densenet import DenseNet121
from models.wideresnet import WideResNet40

from models.flows import Flows, LogLikelihood

from datasets.dataset import CustomDataset

from utils.config import get_args
from utils.general import  init_experiment, init_classifier, init_test, save_test
from utils.score import get_mahalanobis_score, get_odin_score, sample_estimator, get_measures, print_measures, get_gem_score


def run(**kwargs):
    state = init_experiment({key: value for key, value  in kwargs.items()}, 'test')
    dset = CustomDataset(state)

    cls = None 
    if state['classifier_model'] == 'wideresnet40':
        cls = WideResNet40(dset.num_classes) 
    elif state['classifier_model'] == 'mobilenetv2':
        cls = MobileNetV2(dset.num_classes)
    elif state['classifier_model'] == 'resnet18':
        cls = ResNet18(dset.num_classes)
    elif state['classifier_model'] == 'densenet121':
        cls = DenseNet121(dset.num_classes)
    state, cls, device = init_classifier(state, cls)
    
    nfs = Flows(state, cls.dims_out)
    criterion = LogLikelihood(state) 
    state, nfs = init_test(state, nfs, device)


    def get_scores(loader, method, in_dist=False):
        if method == 'Mahalanobis' or method == 'GEM':
            scores = []
            inf_start = tm.time()
            # Get sample mean and precision matrix
            cls.eval()
            temp_x = torch.rand(2,3,32,32).cuda()
            temp_x = Variable(temp_x)
            temp_list = cls.feature_list(temp_x)[1]
            count = len(temp_list)
            feature_list = [out.size(1) for out in temp_list]
            sample_mean, precision = sample_estimator(cls, dset.num_classes, feature_list, dset.train_loader)
            # Get scores
            noise = 0.0005
            if method == 'GEM':
                scores = get_gem_score(cls, loader, dset.num_classes, sample_mean, precision, count-1, noise, dset.num_batches, in_dist=in_dist)
            elif method == 'Mahalanobis':
                scores = get_mahalanobis_score(cls, loader, dset.num_classes, sample_mean, precision, count-1, noise, dset.num_batches, in_dist=in_dist)
            inf_end = tm.time()
            
            return scores, inf_end-inf_start
        else:
            scores = []
            inf_start = tm.time()
            to_np = lambda x: x.data.cpu().numpy()
            concat = lambda x: np.concatenate(x, axis=0)
            for batch_idx, (data, _) in enumerate(loader):
                data = data.cuda()
                if batch_idx >= dset.num_batches and dset.prelims and in_dist is False:
                    break
                if method == 'ODIN':
                    cls.train()
                    data.requires_grad_()
                    logits, _ = cls(data)
                    temperture = 1000.0
                    noise = 0.0048
                    odin_score = get_odin_score(data, logits, cls, temperture, noise)
                    max_odin = np.max(odin_score, 1)
                    scores.append(-max_odin)
                else:
                    with torch.no_grad():
                        if method == 'Energy' or method == 'Energy-FT' or method == 'VOS-FT':
                            cls.train()
                            logits, _ = cls(data)
                            temperture = 1000.0 if method == 'Energy-FT' else 1.0
                            scores.append(-to_np((temperture*torch.logsumexp(logits/temperture, dim=1))))
                        elif method == 'MSP' or method == 'OE-FT':
                            cls.train()
                            logits, _ = cls(data)
                            smax = to_np(F.softmax(logits, dim=1))
                            scores.append(-np.max(smax, axis=1))
                        elif method == 'QuantOD':
                            cls.eval()
                            nfs.eval()
                            _, features = cls(data)
                            z, ljd = nfs(features)
                            ll, _ = criterion(z, ljd)
                            scores.append(to_np(ll))
                        else:
                            raise ValueError('Method not supported: {}'.format(method))
            inf_end = tm.time()
            if in_dist:
                return concat(scores).copy(), inf_end-inf_start
            else:
                if dset.prelims:
                    return concat(scores)[:dset.num_samples].copy(), inf_end-inf_start
                else:
                    return concat(scores).copy(), inf_end-inf_start


    def calc_performance(in_score, id_inf, out_score, ood_inf, ood, state, method, exp):
        # out_score, ood_inf = get_scores(ood_loader, method)
        inf = id_inf+ood_inf
        auroc, aupr, fpr = 0, 0, 0
        if method == 'QuantOD':
            auroc, aupr, fpr = get_measures(in_score, out_score)
        elif method == 'OE-FT':
            auroc, aupr, fpr = get_measures(out_score, in_score)
        else:
            auroc, aupr, fpr = get_measures(-in_score, -out_score)

        log = {'method': method, 'cls':state['classifier_model'], 
               'ind': state['id_dataset'], 'ood': ood, 'exp': exp, 
               'fpr95': round(fpr*100, 2), 'auroc': round(auroc*100, 2),
               'aupr': round(aupr*100, 2), 'infr': round(inf, 2)}
        
        return auroc, aupr, fpr, inf, log
    

    def run_experiments():
        experiments = []
        loaders = [(dset.ood_loader, state['ood_dataset'])] if state['ood_dataset'] != 'all' else dset.ood_loaders
        for method in ast.literal_eval(state['evalutaion_methods']):
            for ood_loader, ood in loaders:
                aurocs, auprs, fprs, infs = [], [], [], []
                for exp in range(state['experiment_count']):
                    id_score, id_inf = get_scores(dset.test_loader, method, in_dist=True)
                    out_score, ood_inf = get_scores(ood_loader, method)
                    auroc, aupr, fpr, inf, log = calc_performance(id_score, id_inf, out_score, ood_inf, ood, state, method, exp)
                    aurocs.append(auroc); auprs.append(aupr); fprs.append(fpr); infs.append(inf); experiments.append(log)
                    
                auroc_avg = np.mean(aurocs); aupr_avg = np.mean(auprs); fpr_avg = np.mean(fprs); inf_avg = np.mean(infs)
                print_measures(auroc_avg, aupr_avg, fpr_avg, inf_avg, f"{method}: {ood.upper()}")

        df_results = pd.DataFrame(experiments)
        save_test(state, df_results)
    
    # Run experiments
    run_experiments()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = get_args()
    main(opt)
