"""Learned macro plan and four-layer compound action query decoder."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .action_model import ActionDirector,position
from .choreography import PLAN_SIZE,PARAM_SIZE,MATERIALS,VERBS,CHANNELS

MAX_ACTIONS=64


class ChoreographyStudent(ActionDirector):
    def __init__(self,vocabulary,width=256,plan_memory=True):
        super().__init__(vocabulary,width,visibility_head=True,direct_effects=True)
        self.structure=nn.Linear(width,PLAN_SIZE)
        self.plan_encoder=nn.Sequential(nn.Linear(PLAN_SIZE,width),nn.GELU(),nn.Linear(width,width))
        self.queries=nn.Embedding(MAX_ACTIONS,width)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,8,width*4,.1,batch_first=True,norm_first=True),4)
        self.action_count=nn.Linear(width,MAX_ACTIONS+1)
        self.material_head=nn.Linear(width,len(MATERIALS))
        self.verb_head=nn.Linear(width,len(VERBS))
        self.object_head=nn.Linear(width,8)
        self.parameter_head=nn.Linear(width,PARAM_SIZE+2)
        self.gradient_checkpointing=True
        self.plan_memory=plan_memory

    def encode(self,batch,use_teacher=False):
        whole=batch['whole'];width=self.global_in.out_features
        g=self.global_in(whole)+position(torch.arange(whole.shape[1],device=whole.device)[None]*2,width)
        g=self.planner(g,src_key_padding_mask=batch['whole_mask'])
        structure_logits=self.structure(g)
        structure=structure_logits.sigmoid()
        plan=batch['whole_plan'] if use_teacher else structure
        indices=batch['indices'].clamp(max=g.shape[1]-1)
        context=g.gather(1,indices[...,None].expand(-1,-1,width))
        p=plan.gather(1,indices[...,None].expand(-1,-1,PLAN_SIZE))
        h=self.local_in(batch['audio'])+context+self.plan_encoder(p)+position(batch['times'],width)
        h=self.local(h,src_key_padding_mask=~batch['mask'])
        return h,g,structure,structure_logits

    def forward_actions(self,batch,use_teacher=False):
        h,g,structure,structure_logits=self.encode(batch,use_teacher)
        weights=batch['mask'][...,None];pooled=(h*weights).sum(1)/weights.sum(1).clamp_min(1)
        q=self.queries.weight[None].expand(len(h),-1,-1)+pooled[:,None]
        memory=h;memory_mask=~batch['mask']
        if self.plan_memory:
            raw_plan=batch['whole_plan'] if use_teacher else structure
            indices=batch['indices'].clamp(max=raw_plan.shape[1]-1)
            tokens=self.plan_encoder(raw_plan.gather(1,indices[...,None].expand(-1,-1,PLAN_SIZE)))
            memory=torch.cat([h,tokens*2.],dim=1);memory_mask=torch.cat([memory_mask,memory_mask],dim=1)
        if self.training and self.gradient_checkpointing:
            for layer in self.decoder.layers:
                q=checkpoint(lambda x,m,layer=layer:layer(x,m,memory_key_padding_mask=memory_mask),q,memory,use_reentrant=False)
        else:q=self.decoder(q,memory,memory_key_padding_mask=memory_mask)
        logits=self.token(h);residual=.25*self.residual(h).tanh()
        motion=(logits.softmax(-1)@self.vocabulary+residual).clamp(0,1)
        motion=motion.clone();motion[...,[0,7]]=self.visibility(h).sigmoid()
        return dict(structure=structure,structure_logits=structure_logits,plan=self.plan_head(g).sigmoid(),logits=logits,residual=residual,
            motion=motion,accent=self.accent(h).squeeze(-1),visibility=self.visibility(h),effects=self.effects(h),
            count=self.action_count(pooled),material=self.material_head(q),verb=self.verb_head(q),
            object=self.object_head(q),parameters=self.parameter_head(q).sigmoid())


def student_loss(pred,batch):
    mask=batch['mask'];valid=mask.sum().clamp_min(1)
    motion=F.smooth_l1_loss(pred['motion'],batch['motion'],reduction='none').mean(-1)
    accent=F.binary_cross_entropy_with_logits(pred['accent'],batch['accent'],reduction='none',pos_weight=pred['accent'].new_tensor(2.))
    loss=((12*motion+accent)*mask).sum()/valid
    pair=mask[:,1:]&mask[:,:-1]
    velocity=F.smooth_l1_loss(torch.diff(pred['motion'],dim=1),torch.diff(batch['motion'],dim=1),reduction='none').mean(-1)
    loss+=2*(velocity*pair).sum()/pair.sum().clamp_min(1)
    synthetic=batch['synthetic_weight']
    structural=F.binary_cross_entropy_with_logits(pred['structure_logits'],batch['whole_plan'],reduction='none').mean(-1)
    whole_weights=batch.get('whole_weights',synthetic[:,None])
    structural=(structural*(~batch['whole_mask'])*whole_weights).sum(1)/(~batch['whole_mask']).sum(1).clamp_min(1)
    loss+=structural.mean()*2
    counts=F.cross_entropy(pred['count'],batch['count'],reduction='none')
    eventloss=pred['count'].new_zeros(len(synthetic))
    event_weights=batch.get('event_weights',synthetic[:,None])*batch['event_mask']
    for key in ('material','verb','object'):
        field=F.cross_entropy(pred[key].transpose(1,2),batch[key],reduction='none')
        eventloss+=(field*event_weights).sum(1)/batch['event_mask'].sum(1).clamp_min(1)
    param=F.smooth_l1_loss(pred['parameters'],batch['parameters'],reduction='none')
    param=(param*batch['parameter_mask']).sum(-1)/batch['parameter_mask'].sum(-1).clamp_min(1)
    eventloss+=16*(param*event_weights).sum(1)/batch['event_mask'].sum(1).clamp_min(1)
    loss+=(counts*synthetic+eventloss).mean()*4
    return loss
