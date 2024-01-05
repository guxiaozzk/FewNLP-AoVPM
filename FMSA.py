import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM,AutoTokenizer,RobertaTokenizer,RobertaForMaskedLM
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timm
from typing import Any
torch.set_printoptions(threshold=np.inf)
#VQA 协同注意力机制
from typing import Dict, Optional
 
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
def Attention( query_local, key_local):
#             batch_size_new = value_local.size(0)
#             h_local, w_local = value_local.size(1), value_local.size(2)
#             value_local = value_local.contiguous().view(batch_size_new, self.in_dim, -1)
#             print('self-attention before',value_local.shape)
#             query_local = query_local.contiguous().view(batch_size_new, self.in_dim, -1)
    query_local = query_local.permute(0, 2, 1)
#             key_local = key_local.contiguous().view(batch_size_new, self.in_dim, -1)
#             print(query_local.shape,key_local.shape)
    sim_map = torch.bmm(key_local, query_local)



    return sim_map

def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
 
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()
 
 
def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
#     print(src_lengths)
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # print('bsz:', bsz)
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)
 
    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
 
class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking
 
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, Q_lengths):
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        # (batch_size, seq_len, region_num)
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
 
        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
        # # (batch_size, 1, seq_len)
 
        masked_a_q = masked_softmax(
            a_q.squeeze(1), Q_lengths, self.src_length_masking
        ).unsqueeze(1)
 
        # (batch_size, hidden_dim)
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_dim)
        q = torch.squeeze(torch.matmul(masked_a_q, Q))
 
        return a_v, masked_a_q, v, q

class FocalLoss(nn.Module):
    def __init__(self, gamma = 1., alpha = 1., size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
        
import torch
from torch import nn
from torch.nn import Module, Conv2d, Sigmoid, Tanh, ModuleList


class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
#         print(self.U.shape)
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
#         print('bilinar_mapping',bilinar_mapping.shape)
        return bilinar_mapping

class myModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dropout_rate: 0.0):
        super().__init__()
       
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*1024, out_features=128),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*1024, out_features=128),
                                            torch.nn.ReLU())
        self.biaffne_layer = biaffine(128,2)

        self.lstm=torch.nn.LSTM(input_size=1024,hidden_size=1024, \
                        num_layers=1,batch_first=True, \
                        dropout=0.5,
                        # dropout=0.1,
                        bidirectional=True)
        self.trasform = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8, dropout=0.1)  # 这里用了八个头
        self.transformer_encoder = nn.TransformerEncoder(self.trasform, num_layers=2)

        
        self.relu=torch.nn.ReLU()
#         self.logit = nn.Linear(7*7,1024)
        self.logits_layer=torch.nn.Linear(in_features=1024, out_features=2)
        

    def forward(self,input,inputs,is_training=True):
#         bert_output = self.roberta_encoder(input_ids=input['input_ids'], 
#                                             attention_mask=input['attention_mask'], 
#                                             token_type_ids=input['token_type_ids']) 
#         encoder_rep = bert_output[0]
#         bert_output2 = self.roberta_encoder(input_ids=inputs['input_ids'], 
#                                             attention_mask=inputs['attention_mask'], 
#                                             token_type_ids=inputs['token_type_ids']) 
#         encoder_rep2 = bert_output2[0]
        encoder_rep = input 
        encoder_rep2= inputs
#         print('encoder_rep',encoder_rep.shape)
        encoder_rep,_ = self.lstm(encoder_rep)
        
        encoder_rep2,_ = self.lstm(encoder_rep2)
        # encoder_rep = self.transformer_encoder(encoder_rep)
        # encoder_rep2 = self.transformer_encoder(encoder_rep2)

        start_logits = self.start_layer(encoder_rep) 
        end_logits = self.end_layer(encoder_rep2) 
        # print("start_logits.shape",start_logits.shape)
        span_logits = self.biaffne_layer(start_logits,end_logits)
        # print("out of biaff",span_logits.shape) 
#         1,len,len,4
        # span_logits = span_logits.reshape(span_logits.shape[0],span_logits.shape[1]*span_logits.shape[2]*span_logits.shape[3])
        ##形状不变
        # print("out contigupus",span_logits.shape)
        # span_logits = self.relu(span_logits)
        # span_logits = self.logits_layer(span_logits)

        span_prob = torch.nn.functional.softmax(span_logits, dim=-1)
        # span_prob =  nn.Linear(span_prob.shape[1],1024)(span_prob)
        ##全部拍成1 因为 拍成别的太大了  循环拍成1试试
        if is_training:
            return span_logits
        else:
            return span_prob



class PromptEncoder(nn.Module):
    '''learnable token generator modified from P-tuning
    https://github.com/THUDM/P-tuning
    '''
    def __init__(self, prompt_token_len, hidden_size, device, lstm_dropout,args,label_id_list):
        super().__init__()
        print("[#] Init prompt encoder...")
        # Input [0,1,2,3,4,5,6,7,8]
        self.seq_indices = torch.LongTensor(list(range(prompt_token_len))).cuda()
        # Embedding
        self.embedding = nn.Embedding(prompt_token_len, hidden_size)
        self.out_embedding = nn.Embedding(31112,hidden_size)
        # self.out_embedding = nn.Embedding(119548,hidden_size)
        # LSTM
        self.final_lstm = nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.lstm_head = nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size))
        self.device = device
        self.args = args
        self.label_id_list = label_id_list
        self.hidden_size=hidden_size
        self.query_lin = nn.Linear(1024,1024)
        self.key_lin = nn.Linear(1024,1024)
        self.value_lin = nn.Linear(1024,1024)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        # self.mlp = nn.Linear(9,7).to(device)
        self.gcn = GCN(1024,1024,3)
        embed_dim = 1024
        self.num_heads = 1
        # 输出是 (attn_output, attn_output_weights)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, self.num_heads)
        self.baf_model = myModel(dropout_rate=0.0)
        self.affine1 = nn.Parameter(torch.Tensor(1024,1024))
        self.affine2 = nn.Parameter(torch.Tensor(1024, 1024))
        self.out_lin = nn.Linear(1024*3,1024)
        self.final_mlp = nn.Linear(4,12)
    def get_att_score(self,lstm_out,or_text):
        query = self.query_lin(lstm_out).transpose(0,1)
        key = self.key_lin(lstm_out).transpose(0,1)
        value =  self.value_lin(lstm_out).transpose(0,1)
        # print(query.device)
        att_score = self.multihead_attn(query, key, value)[1]
        # print(att_score.shape)
        attn_adj_list = att_score
        # attn_adj_list = attn_adj_list.squeeze(0)
        
        # attn_adj_list1=attn_adj_list.clone()
        # print(attn_adj_list)
        mask_ = (torch.zeros_like(or_text) != or_text).float().unsqueeze(-1)[:,:]
        adj_ag = None
        # * Average Multi-head Attention matrixes
        for i in range(1):
            if adj_ag is None:
                adj_ag = attn_adj_list
            else:
                adj_ag += attn_adj_list[i]
        adj_ag /= self.num_heads
        # print(adj_ag.device)
        # adj_ag = adj_ag.squeeze(0)
        # print(adj_ag.size(0),adj_ag.size(1))
        for j in range(adj_ag.size(0)):
        #     print(adj_ag[j].shape)
        #     print(torch.diag(torch.diag(adj_ag[j])))
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            # print(torch.eye(adj_ag[j].size(0)).cpu())
            # print(type(adj_ag))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()

        adj_ag = mask_ * adj_ag
        return adj_ag

    def attention_net2(self,lstm_output, final_state,l_tem,f_tem,mask=None,device=None):
#         print(lstm_output.shape,l_tem.shape)

        
        ##get baffine attention out
        
#         print(baff_out.shape,lstm_output.shape)
#         out = torch.cat((lstm_out,baff_out.unsqueeze(1).to('cuda')),dim=1)
#         print(out.shape)
#         out = nn.Linear(out.shape[0], input_embeds.shape[1])
    
        #batch_size=1,n_step=length
        ##文本的Q和K是相同的
        # mask = np.triu(torch.ones(batch, n_q, n_k), k=1)
        # mask = torch.from_numpy(mask).byte()
        # baff_out = myModel(dropout_rate=0.0)(lstm_output.to('cpu'),lstm_output.to('cpu'))
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # print(final_state.shape)
        lstm_output_o = lstm_output.clone()
        final_state_o = final_state.clone()
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
        lstm_output,final_state = lstm_output[:,:9,:],final_state
        # print(final_state.shape)
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context_1 = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
        lstm_output,final_state =  lstm_output_o[:,9:,:],final_state_o
        # print(final_state.shape)
        steps = lstm_output.shape[1]
        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0],final_state[1]),dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        outputs=[]
        # attn_weights : [batch_size,n_step]
#         for i in range(steps):

#         if mask is not None:
#             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        lstm_out = lstm_output.transpose(1,2)
#             self.lstm_head()

        context_2 = torch.bmm(lstm_out,soft_attn_weights.unsqueeze(2)).squeeze(2)
#         print(lstm_out.shape,context.unsqueeze(2).shape)  
#             1,1024
#         batch_size = len(l_tem)
#         # hidden = final_state.view(batch_size,-1,1)
#         hidden = torch.cat((f_tem[0],f_tem[1]),dim=1).unsqueeze(2)
#         # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
#         attn_weights = torch.bmm(l_tem, hidden).squeeze(2)
#         outputs=[]
#         # attn_weights : [batch_size,n_step]
# #         for i in range(steps):

# #         if mask is not None:
# #             attn_weights = attn_weights.masked_fill(mask, -np.inf)  # 3.Mask

#         soft_attn_weights = F.softmax(attn_weights,1)

#         # context: [batch_size, n_hidden * num_directions(=2)]
#         l_tem = l_tem.transpose(1,2)
# #             self.lstm_head()

#         context_tem = torch.bmm(l_tem,soft_attn_weights.unsqueeze(2)).squeeze(2)
#         print('compare two shape',baff_out.shape,context.shape,lstm_out.shape)
        
        # lstm_out =torch.cat((lstm_out,context.unsqueeze(2),baff_out.unsqueeze(2).to(device)),dim=2)
        # print('lstm_beforatt',lstm_out.shape)
        # lstm_out1 = lstm_out
        steps = lstm_output_o.shape[1]
        # lstm_out = lstm_output_o.transpose(1,2)
        # lstm_out =torch.cat((lstm_out,context_1.unsqueeze(2),context_2.unsqueeze(2)),dim=2)
        # pcan = ParallelCoAttentionNetwork(1024, 2)
        q_lens = torch.LongTensor([12])
        # att_v, att_q, newv, newq = pcan(v, q, q_lens)
        
        lstm_out =torch.cat((lstm_out,context.unsqueeze(2)),dim=2)

        # lstm_out =torch.cat((lstm_out,baff_out.unsqueeze(2).to(device)),dim=2)
        # print('lstm_foratt',lstm_out.shape)
        
        # baff_out = myModel(dropout_rate=0.0)(lstm_out.transpose(1,2).to('cpu'),lstm_out.transpose(1,2).to('cpu'))
        # print(baff_out.shape)
        # lstm_out =torch.cat((lstm_out,baff_out.unsqueeze(2).to(device)),dim=2)

        # mlp = nn.Linear(lstm_out.shape[2],steps).to(device)



        #         print(type(lstm_out))
        context = self.final_mlp(lstm_out)
        context  = context.transpose(1,2).squeeze(0)
#             outputs.append(context)
        return context, soft_attn_weights


    def forward(self):
#         print(input_embeds.shape)
        template = "it expresses [MASK] emotion" 
        # template = "it expresses positive emotion" 
        input_templatembeds = self.tokenizer(template,return_tensors='pt')['input_ids'].to(self.device)
#         print(input_templatembeds.shape[1])
        input_templates = self.out_embedding(input_templatembeds)
        # print(input
        # print(type(self.seq_indices))
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        
        # print(h_n.shape)
        lstm_out2,(f_tem,c_tem) = self.lstm_head(torch.cat((input_embeds,input_templates),dim=1))
        # print(h_n.shape)
        #         print()

        lstm_out,(h_n,c_n) = self.lstm_head(input_embeds)
        
        l_tem,(f_tem,c_tem) = self.lstm_head(input_templates)
        
#         print()
        step=lstm_out.shape[1]
       
        # baf_model.train()
        # baff_out = baf_model(l_tem.to('cpu'),l_tem.to('cpu'))
        # baff_adj = torch.argmax(baff_out,dim=3)
        
        
       
        # baf_model.eval()
        # baff_out = baf_model(lstm_out.to('cpu'),lstm_out.to('cpu'))
        baff_out = self.baf_model(lstm_out2,lstm_out2)
        # print(baff_out.shape)
        baff_adj = torch.argmax(baff_out,dim=3)
        
     

        # print('lstm_out',lstm_out.shape)
        adj_ag = self.get_att_score(lstm_out,self.seq_indices.unsqueeze(0)).squeeze(0)
        adj_ag1 = self.get_att_score(lstm_out[:,:9,:],self.seq_indices.unsqueeze(0)[:,:9]).squeeze(0)
        # print(adj_ag1.shape)
        adj_ag2 = self.get_att_score(lstm_out[:,9:,:],self.seq_indices.unsqueeze(0)[:,9:]).squeeze(0)
        # print(adj_ag2.shape)
        adj_agsep = torch.zeros(1,9,9)
    #     print(adj_agsep.shape)
           
        for j in range(adj_ag1.size(1)):
             for i in range(adj_ag1.size(1)):

        #     print(adj_ag[j].shape)
        #     print(torch.diag(torch.diag(adj_ag[j])))
                adj_ag[j][i] += adj_ag1[j][i]
        for j in range(adj_ag2.size(1)):
            for i in range(adj_ag2.size(1)):

        #     print(torch.eye(adj_ag[j].size(0)).cpu())
                adj_ag[j+adj_ag1.size(1)][i+adj_ag1.size(1)] += adj_ag2[j][i]

    #     print(add_ag)
#     #     adj_ag = add_ag+adj_ag
        adj_ag=adj_ag.unsqueeze(0)

        # print(adj_ag.shape)
        out_g = self.gcn(lstm_out,adj_ag)
        # print(out_g.shape)
        out_baff = self.gcn(lstm_out2,baff_adj)[:,:step,:]
        

        A1 = F.softmax(torch.bmm(torch.matmul(out_baff, self.affine1), torch.transpose(out_g, 1, 2)), dim=-1)
        A2 = F.softmax(torch.bmm(torch.matmul(out_g, self.affine2), torch.transpose(out_baff, 1, 2)), dim=-1)
        dep_out, ag_out = torch.bmm(A1, out_g), torch.bmm(A2, out_baff)
        lstm_out = self.attention_net2(lstm_out,h_n,None,None,device=self.device)[0].unsqueeze(0)
        # print(lstm_out.shape)
        
        out2 = torch.cat((lstm_out,out_g,out_baff),dim=2)
        # print(out2.shape)
        out2 = self.out_lin(out2)
        # print(out2.shape)
        output_embeds = self.mlp_head(out2).squeeze()
        # print(output_embeds.shape)
        baff_out = self.baf_model(l_tem,l_tem)
        return output_embeds,baff_out


class VisualEncoder(nn.Module):
    def __init__(self, model_name, img_token_len, embedding_dim):
        super().__init__()
        self.is_resnet = False
        self.img_token_len = img_token_len
        self.embedding_dim = embedding_dim
        from timm.models.efficientnet import _cfg

        config = _cfg(url='', file='/home/zzk/snap/snapd-desktop-integration/57/Downloads/nf_resnet50_ra2-9f236009.pth') #file为本地文件路径

        self.backbone = timm.create_model(model_name,  pretrained=True,
#                                 features_only=True,
                                pretrained_cfg=config)
#         print(self.backbone.head)
        if "resnet" in model_name:
            self.is_resnet = True
            if model_name == "resnet50":
                self.global_pool = self.backbone.global_pool
            else:
                self.global_pool = self.backbone.head.global_pool
            self.visual_mlp = nn.Linear(2048, img_token_len * embedding_dim)  # 2048 -> n * 1024
        elif "vit" in model_name:
            self.visual_mlp = nn.Linear(1024, img_token_len * embedding_dim)  # 1024 -> n * 1024
        
    def forward(self, imgs_tensor):
        bs = imgs_tensor.shape[0]
        # print('img bs',bs)
        visual_embeds = self.backbone.forward_features(imgs_tensor)
        if self.is_resnet:
            visual_embeds = self.global_pool(visual_embeds).reshape(bs, 2048)
        # print(visual_embeds.shape)
#         with open('orginalimgrepresent.txt','w') as f:
# # # #                 print(it)
#                 f.writelines(str(visual_embeds))
        visual_embeds = self.visual_mlp(visual_embeds)
        visual_embeds = visual_embeds.reshape(bs, self.img_token_len, self.embedding_dim)
        # print(visual_embeds.shape)
        return visual_embeds

class GCN(nn.Module):
    def __init__(self,embed_dim,mem_dim, num_layers):
        super(GCN, self).__init__()
#         self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = embed_dim
#         self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(1024, 1024, 2, batch_first=True, \
                dropout=0.0, bidirectional=False)
#         if False:
#             self.in_dim = rnn_hidden * 2
#         else:
#             self.in_dim = 

        # drop out
        self.rnn_drop = nn.Dropout(0.25)
        self.in_drop = nn.Dropout(0.25)
        self.gcn_drop = nn.Dropout(0.25)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attention_heads = 1
        self.head_dim = self.mem_dim // self.layers
#         self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)
        self.weight_list = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, 1024, 2, False)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs,out_adj):
        embs= inputs     # unpack inputs
#         print(type(embs))
#         embs = self.in_drop(embs)

#         # rnn layer
#         print(embs.shape)
        self.rnn.flatten_parameters()
#         gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, embs.shape[1],1))
        gcn_inputs = embs

#         print(gcn_inputs.shape)
#         denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        adj_ag  = out_adj.float()
        outputs = gcn_inputs
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
#         print(outputs.shape)
        for l in range(self.layers):
            Ax = adj_ag.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return outputs
class MSAModel(torch.nn.Module):
    '''main model
    '''
    def __init__(self, args, label_id_list):
        super().__init__()
        self.args = args
        self.label_id_list = label_id_list
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        # self.lm_model =RobertaForMaskedLM.from_pretrained(args.model_name)
        # self.embeddings = self.lm_model.roberta.get_input_embeddings()

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)

        
        self.lm_model = BertForMaskedLM.from_pretrained(args.model_name)

        self.embeddings = self.lm_model.bert.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim  # 1024
        self.para_forloss = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.lin = torch.nn.Linear(13,12)
        self.linvis = torch.nn.Linear(2,1)
        self.pcan = ParallelCoAttentionNetwork(1024, 1024)
        if not args.no_img:
            
            self.img_token_id = self.tokenizer.get_vocab()[args.img_token]
            self.img_token_len = args.img_token_len
            self.visual_encoder = VisualEncoder(args.visual_model_name, self.img_token_len, self.embedding_dim)
            

        if args.template == 3:
            self.prompt_token_id = self.tokenizer.get_vocab()[args.prompt_token]
            self.prompt_token_len = sum([int(i) for i in args.prompt_shape.split('-')[0]]) + int(args.prompt_shape[-1])
            self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, args.device, args.lstm_dropout,args,self.label_id_list)

    def embed_input(self, input_ids, imgs=None):
        bs = input_ids.shape[0]
        embeds = self.embeddings(input_ids)

        if self.args.template == 3:
            prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
#             print('prompt_positiion',prompt_token_position)
            ##后几位不固定是因为句长不一样 获得到位置可以用来观察输出logit
            visual_embeds = self.visual_encoder(imgs)
            
            q_lens = torch.LongTensor([12]).cuda()
            prompt_embeds = self.prompt_encoder()[0].unsqueeze(0)
            # print(prompt_embeds.shape)
            
            prompt_embeds = prompt_embeds.repeat(bs,1,1)
            # print(bs)
            # print(prompt_embeds.shape)
            prompt_embeds =prompt_embeds
            visual_embeds = visual_embeds.permute(0,2,1)
            co_embeds = self.pcan(visual_embeds, prompt_embeds, q_lens)[3].unsqueeze(dim=1)
            visual_embeds = visual_embeds.permute(0,2,1)
            # prompt_embeds = torch.cat((prompt_embeds,co_embeds),dim=1)
            # print(visual_embeds.shape,covisual_embeds.shape)
            prompt_embeds = torch.cat((prompt_embeds,co_embeds),dim=1)
            # visual_embeds  = self.linvis(torch.cat((visual_embeds,covisual_embeds.cuda()),dim=1).permute(0,2,1)).permute(0,2,1)
            prompt_embeds = self.lin(prompt_embeds.permute(0,2,1))
            prompt_embeds =prompt_embeds.permute(0,2,1)
            # print(prompt_embeds.shape)
            for bidx in range(bs):
                for i in range(self.prompt_token_len):
                    # embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]
                    embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[bidx,i, :]
           
#         if self.args.template == 3:
#             print(input_ids.shape)
#             prompt_token_position = input_ids[:,:self.prompt_token_len+10-1].reshape((bs, self.prompt_token_len+10, 2))[:, :, 1]
#             prompt_embeds = PromptEncoder(self.prompt_token_len+10, self.embedding_dim, args.device, args.lstm_dropout)

#             for bidx in range(bs):
#                 for i in range(self.prompt_token_len+10):
#                     embeds[bidx, i, :] = prompt_embeds[i, :]
        
        
        if not self.args.no_img:
            visual_embeds = self.visual_encoder(imgs)
            # print("visual_embed",visual_embeds.shape)
#           
            img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
            for bidx in range(bs):
                for i in range(self.img_token_len):
                    embeds[bidx, img_token_position[bidx, i], :] = visual_embeds[bidx, i, :]
        
        return embeds
    def get_top_n_presention(self,n,lis):
        global dic 
        dic = self.tokenizer.get_vocab()
        lis1 = sorted(lis,reverse=True)
        newlis=[]
        for i in lis1[:n]:
            newlis.extend( [k for k,v in dic.items() if v == lis.index(i)])
        return newlis
    def forward(self, input_ids, attention_mask, labels,imgs=None):
        # print(labels[labels!=-100])
        inputs_embeds = self.embed_input(input_ids, imgs)
        # print(input_ids.shape)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        output = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits
        l_logits = []      
         #观察soft  prompt 位置的logit评分
        bs =  input_ids.shape[0]
        # print(labels[labels!=-100][0])
        # print("每个批次的大小 bs",bs)
        p_logits=[]
        # prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
        # #选一行position
        # prompt_token_position = prompt_token_position[0]

# #         print(logits.shape)
            
#         for id in range(len(prompt_token_position)):
#             p = logits[0,prompt_token_position[id],:]
# #             print((p))
# #             print(list(p))
# #             p = self.get_top_n_presention(5,list(p)
#             p_logits.append(p)
# #             f.writelines(str(p_logits))
# #         print(p_logits)
#         p_logits = torch.stack(p_logits)
#         p_logits = torch.tensor(p_logits)
        # print(list(p_logits.to('cpu')))

        
#         print(p_logits)
        img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
        img_token_position = img_token_position
        
#         print(logits.shape)
        img_out=[]
        # print(img_token_position.shape)##32,3
        # print(img_token_position
        for it in range(len(img_token_position)):
            img_logits=[]
            for id in range(len(img_token_position[it])):
                p = logits[it,img_token_position[it,id],:]
    #             print((p))
    #             print(list(p))
    #             p = self.get_top_n_presention(5,list(p)
                img_logits.append(p)
#             f.writelines(str(p_logits))
#         print(p_logits)
            img_logits = torch.stack(img_logits)
            img_out.append(img_logits)
        img_out = torch.stack(img_out)
        # print('img_out',img_out.shape)
#         with open('imgrepresent.txt','w') as f:

#             for it in list(p_logits):
# #                 print(it)
#                 f.writelines(str(it))
        # print(logits.shape)
        logits = logits[labels != -100]
        # print(logits.shape)


        for label_id in range(len(self.label_id_list)):
            l_logits.append(logits[:, self.label_id_list[label_id]].unsqueeze(-1))
        l_logits = torch.cat(l_logits, -1)
        baff_out = self.prompt_encoder()[1].cpu()

#         print(l_logits.shape)
        probs = l_logits
#         print(probs)
        loss_func = FocalLoss()
#         print(logits.shape)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
#         print(logits.shape,labels[labels!=-100].shape)
       
#         print(labels)
        # print(labels[labels != -100])
#         print(inputs_embeds,labels,logits)
#         pred = logits[labels != -100]
#         print(pred.shape)
#         probs = pred[:, self.label_id_list]
#         print(probs,labels[labels != -100])
        alpha = torch.nn.Parameter(torch.tensor([0.005]), requires_grad=True).to(self.args.device)
        test_tensor=torch.Tensor([[0,0,0,0,0,0],
                                  [0,0,1,0,0,0],
                                  [0,1,0,1,1,0],
                                  [0,0,1,0,1,0],
                                  [0,0,1,1,0,0],
                                  [0,0,0,0,0,0]])
        # print(baff_out.shape)
        dep_loss = loss_fct(baff_out.squeeze(0).reshape(-1,2),test_tensor.reshape(-1).long())
        # print('dep_loss',dep_loss)
        lab = labels[labels != -100]
        # print(lab.shape)
        loss = loss_fct(logits.view(-1, self.lm_model.config.vocab_size),labels[labels!=-100].view(-1))
        le.fit(lab.to('cpu'))
#         mlp = nn.Linear(self.lm_model.config.vocab_size,3).to(self.args.device)
#         pred = mlp(pred)
#         print(lab)
        lab = torch.LongTensor(le.transform(lab.to('cpu')))
#         print(lab.shape)
#         loss=loss_fct(probs.to("cpu"),lab)  

        # foc_loss=loss_func(torch.unsqueeze(probs,2).to("cpu"),torch.unsqueeze(lab ,1))  
        ##除了mask词 其余词损失为null
        # print(probs.shape,type(probs))
        # foc_loss=loss_fct(probs.to("cpu"),lab)  
        # foc_loss = loss_fct(probs.view(-1,3).to('cpu'),lab)
#         print(probs.shape,labels.shape)
#         loss = loss_func(probs,labels)
        # probs_merge = probs[1::2]+probs[::2]
        # all_logits = torch.concat((probs[1::2].unsqueeze(1),probs[::2].unsqueeze(1)),dim=1)
        # # labels = labels[labels != -100][::2]
        # logits_1,logits_2 = probs[1::2],probs[::2]
        # # labels_l = labels[labels != -100]
        # # print(all_logits.shape)
        # fusion_logits =[]
        # # all_label_id = torch.Tensor([0,1,2])
        # all_label_id =[]
        # for label_id in range(len(self.label_id_list)):
        #     all_label_id.append(label_id)
        # all_label_id = torch.tensor(all_label_id)
        # # print(all_label_id)
        # for i in range(len(all_logits)):
        #     new_preds = []
        #     scores = all_logits[i]
        #     for pred_class in all_label_id:
        #         log_scores = torch.log(scores)
        #         sum_logits = torch.sum(log_scores, axis=0)
        #         exp_logits = torch.exp(sum_logits)
        #         # print(exp_logits.shape,pred_class)
        #         out_score = exp_logits[pred_class] / torch.sum(exp_logits)
        #         new_preds.append(out_score)
        #     new_preds = torch.tensor(new_preds)
        #     fusion_logits.append(new_preds)
        #     # print(fusion_logits)
        # fusion_logits = torch.stack(fusion_logits, 0).cuda()
#         print(fusion_logits.shape)
             
#         if False:##两标签二分类
#             # Regression task
#             loss_fct = nn.KLDivLoss(log_target=True)
            
#             labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#             loss_1 = loss_fct(logits_1.view(-1, 2), labels)
#             loss_2 = loss_fct(logits_2.view(-1, 2), labels)
#             loss = (loss_1+loss_2)/2

            # loss.requirezs_grad_(True)
            # print(loss.requires_grad)
#         if True:
# #             print(len(labels))
#             alpha = 0.5
#             loss_fct = nn.CrossEntropyLoss()
#             loss_1 = loss_fct(logits_1.view(-1, logits_1.size(-1)), labels.view(-1))
#             loss_2 = loss_fct(logits_2.view(-1, logits_2.size(-1)), labels.view(-1))
#             loss = loss_1*alpha+(1-alpha)*loss_2

        # print(probs[1::2].shape,probs[::2].shape)
    
        import numpy as np
        probs_merge = probs
        # print(fusion_logits.shape)
        # probs_merge = probs[1::2]+probs[::2]
        # foc_loss = loss_fct(probs_merge.view(-1,3).to('cpu',lab[::2]))
        # probs_merge = fusion_logits
        # print(probs_merge)
        # probs_merge = torch.nn.functional.normalize(probs_merge.cpu(), p=2.0, dim=1, eps=1e-12, out=None)

        # probs_merge = torch.sigmoid(probs_merge)
        # foc_loss = loss_fct(probs_merge.view(-1,3).to('cpu'),lab)
        # print(probs_merge.shape)
        # probs = torch.nn.functional.softmax(probs)#差别在这mvsa-s上不大
        # probs_merge = probs[1::2]+probs[::2]
        pred_labels_idx = torch.argmax(probs_merge, dim=-1).tolist()
        # print(len(pred_labels_idx))
        y_ = [self.label_id_list[i] for i in pred_labels_idx]
        # print(len(y_))
        # y_ = nn.normalize(y_)
       
        # y_ = F.normalize(torch.Tensor(y_[1::2]))+torch.Tensor(F.normalize(y_[::2]))
        y = labels[labels != -100]
        # print(len(y))
        # print(len(y_),y.shape)
        # print(type(loss),type(alpha))
        to_loss = loss+ alpha*dep_loss
        # to_loss = foc_loss
        # print(to_loss)
        # to_loss =loss
        # to_loss =loss+alpha*foc_loss
        # print(y.shape)
        # print(type(y),len(y.tolist()))
        
        # return to_loss, y_[::2], y.tolist()[::2],p_logits,img_out,pro

        # return loss, y_[1::2], y.tolist()[1::2]
        # print(probs_merge.cpu().tolist())
        return to_loss, y_[::2], y.tolist()[::2],p_logits,img_out,probs_merge.cpu().tolist()
