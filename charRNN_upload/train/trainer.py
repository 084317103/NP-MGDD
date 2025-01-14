# coding:latin-1
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from rdkit import Chem
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger
from rdkit.Chem import RDKFingerprint, DataStructs
from rdkit.Chem import QED


class EarlyStoppingMetric:
    def __init__(self, patience=10, delta=0.001, trace_func=print):
        """
        Args:
            patience (int): How long to wait after the last time the metric improved.
            delta (float): Minimum change in the metric to qualify as an improvement.
            trace_func (function): Print function for logging.
        """
        self.patience = patience
        self.delta = delta
        self.trace_func = trace_func
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, current_metric):
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric < self.best_metric + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = current_metric
            self.counter = 0


class CharRNNTrainer(MosesTrainer):
    def __init__(self, config, model=None):
        self.config = config
        self.record = {'loss': [], 'prior_nll': [], 'agent_nll': []}

        if model is not None:  
            self.model = model
            self.prior = model
            self.agent = model
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            self.prior = self.prior.to(self.device)
            self.agent = self.agent.to(self.device)

    def _train_epoch(self, model, tqdm_data, criterion, optimizer=None):
        
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix = {'loss': 0,
                   'running_loss': 0}
        
        for i, (prevs, nexts, lens) in enumerate(tqdm_data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to('cpu') 
            
            outputs, _, _ = model(prevs, lens)
            
            loss = criterion(outputs.view(-1, outputs.shape[-1]),
                             nexts.view(-1))
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() -
                                        postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):
         
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)
        
        
        device = model.device
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.config.step_size,
                                              self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            
            scheduler.step()
            
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))    
            
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            
            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cuda') 
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)
    
    def _reinforce(self, model, logger=None):
         
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                                        optimizer,
                                                        T_max=400,  
                                                        eta_min=1e-6)  
        early_stopping = EarlyStoppingMetric(patience=20, delta=1e-3)
        tqdm_epochs = tqdm(range(self.config.train_epochs), desc='Training')
        for epoch in tqdm_epochs:
            model.train()
            postfix = {'loss': 0}
            samples = []
            
            current_samples, agent_likelihood, action_probs, action_log_probs = self.agent.sample_ahc(self.config.n_batch, 180) 
            samples.extend(current_samples)
            
            collate_fn = self.get_collate_fn(model)

            prevs, nexts, lens = collate_fn(samples)
            

            filename =f"autodl-tmp/charRNN_upload/score_results/iterations_400_charrnn_180_early_diversity/step_{epoch}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['step_id','smiles','valid','qed'])
                scores = []
                diversity_scores = []
                smiles_counts = Counter(samples)
                for s in samples:
                    mol = Chem.MolFromSmiles(s)
                    is_valid = mol is not None
                    qed_value = QED.qed(mol) if is_valid else 0
                    diversity_score = self._calculate_diversity_score(s, samples,smiles_counts)
                    diversity_scores.append(diversity_score)
                    scores.append(qed_value)
                    writer.writerow([epoch, s, is_valid, qed_value])
                
            
            scores = torch.tensor(scores, device=device)
            diversity_scores = torch.tensor(diversity_scores, device=device)
            total_scores = scores + diversity_scores
            total_scores = torch.autograd.Variable(total_scores).cuda()
            agent_likelihood = agent_likelihood
            prior_likelihood = self.prior.likelihood(prevs, nexts, lens)
            loss = self._compute_loss_ahc(prior_likelihood, agent_likelihood, total_scores)
            

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()  
            scheduler.step()
            
            postfix['loss'] = loss.item()
            tqdm_epochs.set_postfix(postfix)
            
            qed_mean = scores.mean().item()
            early_stopping(qed_mean)
            if early_stopping.early_stop:
                print("Early stopping triggered based on QED mean.")
                break
            
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cuda')
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3]+'_{0:03d}.pt'.format(epoch)
                )
                
    def _compute_loss_ahc(self, prior_likelihood, agent_likelihood, scores):
        sigma = 120 
        topk = 0.25 
        augmented_likelihood = prior_likelihood + sigma * scores
        sscore, sscore_idxs = scores.sort(descending=True)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        
        # Update
        self.record['loss'] += list(loss.detach().cpu().numpy())
        self.record['prior_nll'] += list(-prior_likelihood.detach().cpu().numpy())
        self.record['agent_nll'] += list(-agent_likelihood.detach().cpu().numpy())
        # AHC
        loss = loss[sscore_idxs.data[:int(64 * topk)]]
        return loss.mean()
        
    def _calculate_diversity_score(self, smiles, all_smiles, smiles_counts):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        
        fp = Chem.RDKFingerprint(mol)
        
        similarities = []
        for s in all_smiles:
            other_mol = Chem.MolFromSmiles(s)
            if other_mol is not None:
                other_fp = Chem.RDKFingerprint(other_mol)
                similarity = Chem.DataStructs.FingerprintSimilarity(fp,other_fp)
                similarities.append(similarity)
        if similarities:
            diversity_score = 1-max(similarities)
        else:
            diversity_score = 1
        penalty = smiles_counts[smiles] - 1
        diversity_score -= penalty * 0.3
        return diversity_score
        
    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors],
                                 batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors],
                                 batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors],
                                dtype=torch.long, device=device)
            return prevs, nexts, lens

        return collate

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model
        
    def fit_ahc(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None
        self._reinforce(model, logger)
        return model
