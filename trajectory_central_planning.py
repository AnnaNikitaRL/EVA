import logging

def trajectory_central_planning(replay, embedding, value_buffer, num_neighbors, path_length):
    """ trajectory selection and planning 
    
    Estimates for trajectories from replay buffer are calculated as below.
    Parametric calculation is used for off-trajectory value estimates.
    
                       r_t + gamma * VNP(s_{t+1}), if a_t = a,                
    QNP(s_t, a)   = 
                       Q_theta (s_t, a),        otherwise
                       
                       max_a Q_theta (s_t, a), if t = T
    VNP(s_t)       = 
                       max_a QNP(s_t, a),       otherwise       
    
    """
    
    # taking neighbors from the replay buffer. Each index of the returned embedding 
    # will be start of the trajectory
    traj_start_idxs, traj_embeddings = replay.neighbors( embedding, 
                                                          num_neighbors=num_neighbors, 
                                                          return_embeddings=True)
    
    if len(traj_start_idxs) == 0:
        logging.info("no neighbors found")
    else:
        traj_offsets = trajectory_offset(traj_start_idxs, path_length)
        
        # initialisation and preparation
        traj_offsets     = np.array(traj_offsets)
        traj_start_idxs  = np.array(traj_start_idxs)
        max_traj_offsets = np.max(traj_offsets)      
        VNP = torch.zeros( (len(traj_start_idxs),1) )

        # we update values from the end to the beginning of the trajectories
        for t in reversed(range(min(path_length, max_traj_offsets + 1))):

            nonterm      = t <= traj_offsets
            nonterm_idxs = np.where(nonterm)[0]
            nonterm      = torch.BoolTensor(nonterm)

            state_batch = [replay.memory[(traj_start_idxs[traj_number] + t) % replay.capacity].state 
                           for traj_number in nonterm_idxs]
            state_batch = torch.from_numpy(np.stack(state_batch)).to(device)

            # calculation of parametric values of Q function via neural net
            QNP = qnet(state_batch)[0].detach().cpu()
            
            # for all states corresponding to non last length of trajectory (t != T), 
            # we overwrite parametric values of Q function with 
            # non-parametric values of Q function for the action that has been chosen at moment t, a_t. 
            if t < path_length - 1:
                action_batch = [replay.memory[(traj_start_idxs[traj_number] + t) % replay.capacity].action
                                for traj_number in nonterm_idxs]
                action_batch = torch.LongTensor(action_batch).view(-1,1)
                reward_batch = [replay.memory[(traj_start_idxs[traj_number] + t) % replay.capacity].reward 
                                for traj_number in nonterm_idxs]
                reward_batch = torch.FloatTensor(reward_batch).view(-1,1)
                QNP.scatter_(1, action_batch, reward_batch + gamma * VNP[nonterm])

            VNP[nonterm] = torch.max(QNP,dim=1,keepdim=True)[0]
            
        # store resulting Q value into value buffer alongside their embedding
        for traj_number in range(len(traj_start_idxs)):    
            value_buffer.push(traj_embeddings[traj_number], QNP[traj_number])

        # re-build value buffer to be able to find neighbors taking into account updated values
        value_buffer.build_tree()

def trajectory_offset(traj_start_idxs, path_length):
    """finds last index of the trajectory in replay buffer for start indxs of the trajectories"""
    traj_offsets = [] 
    
    # for each index we follow to the next state in replay buffer until path length or terminal state
    for traj_start_idx in traj_start_idxs:     
        for idx in range(path_length):
            if replay.memory[(traj_start_idx + idx) % replay.capacity].next_state is None:
                break
        traj_offsets.append(idx)
        
    return traj_offsets
