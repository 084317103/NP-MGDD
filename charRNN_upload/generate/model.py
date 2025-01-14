import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class CharRNN(nn.Module):

    def __init__(self, vocabulary, config):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size,
                                            padding_idx=vocabulary.pad)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._nll_loss = nn.NLLLoss(reduction="none").to(self._device)
        


    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device=device
        return super().to(device)

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens
    
    def likelihood(self, sequences, targets, lengths, hiddens=None):
        sequences = sequences.to(self.device)
        lengths = lengths.to('cpu')
        x = self.embedding_layer(sequences)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        logits = self.linear_layer(x)
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1,2),targets).sum(dim=1)
        
    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device
                              if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def sample(self, n_batch, max_length=100):
        with torch.no_grad():
            starts = [torch.tensor([self.vocabulary.bos],
                                   dtype=torch.long,
                                   device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long,
                                  device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long,
                             device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device='cpu')
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]
          
                for j, top in enumerate(ind_tops):
                    
                    if not end_smiles_list[j]:
                        
                        top_elem = top[0].item()
                        
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True
                       
                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long,
                                      device=self.device).unsqueeze(1)

            new_smiles_list = [new_smiles_list[i][:l]
                               for i, l in enumerate(len_smiles_list)]
            return [self.tensor2string(t) for t in new_smiles_list]
            
    def sample_ahc(self, n_batch, max_length=100):
        with torch.no_grad():
            nlls = torch.zeros(n_batch, device=self.device)
            starts = [torch.tensor([self.vocabulary.bos],
                                   dtype=torch.long,
                                   device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long,
                                  device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long,
                             device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device='cpu')
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            action_probs = torch.zeros((n_batch, max_length+1), requires_grad=True, device=self.device)
            action_log_probs = torch.zeros((n_batch, max_length+1), requires_grad=True, device=self.device)
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)

                # probabilities
                probs_list = [F.softmax(o, dim=-1) for o in output]
                log_probs_list = [F.log_softmax(o, dim=-1) for o in output]
                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs_list]
                for j, (top, prob, log_prob) in enumerate(zip(ind_tops, probs_list, log_probs_list)):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        action_probs.data[j, i] = prob[0, top_elem]
                        action_log_probs.data[j, i] = log_prob[0, top_elem]
                        
                        nlls[j] -= log_prob[0,top_elem]
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long, device=self.device).unsqueeze(1)
            action_probs = action_probs[:, :max(len_smiles_list)]
            action_log_probs = action_log_probs[:, :max(len_smiles_list)]
            

            new_smiles_list = [new_smiles_list[i][:l]
                               for i, l in enumerate(len_smiles_list)]
            return [self.tensor2string(t) for t in new_smiles_list], nlls, action_probs, action_log_probs
