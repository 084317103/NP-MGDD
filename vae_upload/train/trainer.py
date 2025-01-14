import torch
import torch.optim as optim
from tqdm.auto import tqdm
import csv
from rdkit import Chem
from torch.nn.utils import clip_grad_norm_
from collections import Counter
from moses.interfaces import MosesTrainer
from moses.utils import OneHotVocab, Logger, CircularBuffer
from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer
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

class VAETrainer(MosesTrainer):
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
    def get_vocabulary(self, data):
        return OneHotVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                       for string in data]

            return tensors

        return collate
    '''
    def _reinforce(self, model, logger=None):
        device = model.device
        n_epoch = self._n_epoch()
        optimizer = optim.Adam(self.get_optim_params(model), lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,self.config)
        
        tqdm_epochs = tqdm(range(n_epoch), desc=f"Training Progress", unit="epoch") 
        for epoch in tqdm_epochs:   
            model.train()
            kl_weight = kl_annealer(epoch)
            postfix = {'loss': 0}
            samples, agent_likelihood, action_probs, action_log_probs = self.agent.sample_ahc(self.config.n_batch, 180) #212
       
            collate_fn = self.get_collate_fn(model)
            texts = collate_fn(samples)
           
            filename =f"autodl-tmp/charRNN/score_results/iterations_400_vae_180_diversity/step_{epoch}.csv"
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
                    scores.append(qed_value)
                    diversity_score = self._calculate_diversity_score(s,samples,smiles_counts)
                    diversity_scores.append(diversity_score)
                    writer.writerow([epoch, s, is_valid, qed_value])
               
            scores = torch.tensor(scores).cuda()
            diversity_scores = torch.tensor(diversity_scores).cuda()
            total_scores = scores + diversity_scores
            total_scores = torch.autograd.Variable(total_scores).cuda()
            agent_likelihood = agent_likelihood
            prior_likelihood = self.prior.likelihood(texts)
            loss = self._compute_loss_ahc(prior_likelihood, agent_likelihood, total_scores)
       
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.get_optim_params(model), self.config.clip_grad)
            optimizer.step()
            
       
            postfix['loss'] = loss.item()
            tqdm_epochs.set_postfix(loss=postfix['loss'])
            lr_annealer.step()
            if logger:
                logger.append(postfix)
                logger.save(self.config.log_file) 
           
            if epoch % self.config.save_frequency == 0:
                model = model.to('cuda')
                torch.save(model.state_dict(),
                          self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch))
    '''
    
    def _reinforce(self, model, logger=None):
        device = model.device
        n_epoch = self._n_epoch()
        optimizer = optim.Adam(self.get_optim_params(model), lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,self.config)
        early_stopping = EarlyStoppingMetric(patience=20, delta=1e-3)
        tqdm_epochs = tqdm(range(n_epoch), desc=f"Training Progress", unit="epoch") 
        for epoch in tqdm_epochs:   
            model.train()
            kl_weight = kl_annealer(epoch)
            postfix = {'loss': 0}
            samples, agent_likelihood, action_probs, action_log_probs = self.agent.sample_ahc(self.config.n_batch, 180) #从212改为180
       
            collate_fn = self.get_collate_fn(model)
            texts = collate_fn(samples)
           
            filename =f"autodl-tmp/charRNN/score_results/iterations_400_vae_180_early_diversity/step_{epoch}.csv"
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
                    scores.append(qed_value)
                    diversity_score = self._calculate_diversity_score(s,samples,smiles_counts)
                    diversity_scores.append(diversity_score)
                    writer.writerow([epoch, s, is_valid, qed_value])
               
            scores = torch.tensor(scores).cuda()
            diversity_scores = torch.tensor(diversity_scores).cuda()
            total_scores = scores + diversity_scores
            total_scores = torch.autograd.Variable(total_scores).cuda()
            agent_likelihood = agent_likelihood
            prior_likelihood = self.prior.likelihood(texts)
            loss = self._compute_loss_ahc(prior_likelihood, agent_likelihood, total_scores)
       
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.get_optim_params(model), self.config.clip_grad)
            optimizer.step()
            
            qed_mean = scores.mean().item()
            early_stopping(qed_mean)
       
            postfix['loss'] = loss.item()
            tqdm_epochs.set_postfix(loss=postfix['loss'])
            lr_annealer.step()
            
            if early_stopping.early_stop:
                print("Early stopping triggered based on QED mean.")
                break
            if logger:
                logger.append(postfix)
                logger.save(self.config.log_file) 
           
            if epoch % self.config.save_frequency == 0:
                model = model.to('cuda')
                torch.save(model.state_dict(),
                          self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch))
                          
    def _calculate_diversity_score(self,smiles, all_smiles, smiles_counts):
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
        
        
    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)
        for input_batch in tqdm_data:
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward
            kl_loss, recon_loss = model(input_batch)
            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model),
                                self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f} lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix
    
    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model),
                               lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer,
                                                   self.config)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix = self._train_epoch(model, epoch,
                                        tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           self.config.model_save[:-3] +
                           '_{0:03d}.pt'.format(epoch))
                model = model.to(device)

            # Epoch end
            lr_annealer.step()
    
    
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
        
    def _n_epoch(self):
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult ** i)
            for i in range(self.config.lr_n_restarts)
        )
