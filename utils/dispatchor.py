#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import copy

def dispatch(client_index, global_model, client_models):
    global_w = global_model.state_dict()
    for k in client_index:
        client_models[k] = copy.deepcopy(global_model)
        client_models[k].load_state_dict(global_w)

