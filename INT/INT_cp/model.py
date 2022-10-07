import torch
import torch.nn as nn
import torch.nn.functional as F

def SALS(A, B, C, T_batch, lr, device, reg=1e-3):
    
    # update factors, compute X and update A, B, C
    X = torch.solve(torch.einsum('ijkl,jr,kr,lr->ri',T_batch,A,B,C), (A.T@A)*(B.T@B)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T

    # update alternatively
    A_star = torch.solve(torch.einsum('ijkl,ir,kr,lr->rj',T_batch,X,B,C), (X.T@X)*(B.T@B)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    A = (1-lr) * A + lr * A_star
    B_star = torch.solve(torch.einsum('ijkl,ir,jr,lr->rk',T_batch,X,A,C), (A.T@A)*(X.T@X)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    B = (1-lr) * B + lr * B_star
    C_star = torch.solve(torch.einsum('ijkl,ir,jr,kr->rl',T_batch,X,A,B), (A.T@A)*(X.T@X)*(B.T@B) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    C = (1-lr) * C + lr * C_star

    rec = torch.norm(torch.einsum('ir,jr,kr,lr->ijkl',X,A,B,C) - T_batch) / torch.norm(T_batch)

    return [A, B, C], rec

def GR_SALS(A, B, C, T_batch, lr, device, reg=1e-3):
    GR_coefficient = 1e-1
    N = T_batch.shape[0]
    G = 100 * (torch.eye(N) * (1 / N + 1e-2) - torch.ones(N, N) / N).to(device)
    
    MTTKRP1 = torch.einsum('ijkl,jr,kr,lr->ri',T_batch,A,B,C)
    X_init = torch.solve(MTTKRP1, (A.T@A)*(B.T@B)*(C.T@C) + torch.eye(B.shape[1]).to(device) * (reg))[0].T

    # update factors, compute X and update A, B, C
    X = torch.solve(torch.einsum('ijkl,jr,kr,lr->ri',T_batch,A,B,C) - GR_coefficient * X_init.T @ G, \
             (A.T@A)*(B.T@B)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T

    # update alternatively
    A_star = torch.solve(torch.einsum('ijkl,ir,kr,lr->rj',T_batch,X,B,C), (X.T@X)*(B.T@B)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    A = (1-lr) * A + lr * A_star
    B_star = torch.solve(torch.einsum('ijkl,ir,jr,lr->rk',T_batch,X,A,C), (A.T@A)*(X.T@X)*(C.T@C) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    B = (1-lr) * B + lr * B_star
    C_star = torch.solve(torch.einsum('ijkl,ir,jr,kr->rl',T_batch,X,A,B), (A.T@A)*(X.T@X)*(B.T@B) + torch.eye(A.shape[1]).to(device) * reg)[0].T
    C = (1-lr) * C + lr * C_star

    rec = torch.norm(torch.einsum('ir,jr,kr,lr->ijkl',X,A,B,C) - T_batch) / torch.norm(T_batch)

    return [A, B, C], rec

def ATD(A, B, C, T_batch, T_batch_aug, lr, device, reg=1e-3, epsilon=5e-1):
    
    MTTKRP1 = torch.einsum('ijkl,jr,kr,lr->ri',T_batch,A,B,C)
    MTTKRP2 = torch.einsum('ijkl,jr,kr,lr->ri',T_batch_aug,A,B,C)
    prep = (A.T@A)*(B.T@B)*(C.T@C)
    
    N = T_batch.shape[0]
    
    # cold start
    X_aug_init = torch.solve(MTTKRP2, prep + torch.eye(B.shape[1]).to(device) * (reg))[0].T
    X_init = torch.solve(MTTKRP1, prep + torch.eye(B.shape[1]).to(device) * (reg))[0].T
    
    # auxiliary step
    G = (torch.ones(N, N) / N - torch.eye(N) * (1 / N + 1e-2)).to(device)
    V3 = torch.inverse(prep + torch.eye(B.shape[1]).to(device) * reg)
    result_list = []
    for j in range(1):
        result_list.append(X_init.clone())
        norm_aug = torch.diag(1.0 / (X_aug_init ** 2).sum(dim=1) ** .5)
        norm = torch.diag(1.0 / (X_init ** 2).sum(dim=1) ** .5)
        V2 = norm_aug @ X_aug_init
        V2_aug = norm @ X_init
        for i in range(X_init.shape[1]):
            X_init[i,:] = (MTTKRP1[:, i] - epsilon * norm[i,i] * (G[i,:] @ V2) @ (torch.eye(B.shape[1]).to(device) - \
                        X_init[i,:].T @ X_init[i,:] * norm[i,i]**2)) @ V3
        for i in range(X_init.shape[1]):
            X_aug_init[i,:] = (MTTKRP2[:, i] - epsilon * norm_aug[i,i] * (G[i,:] @ V2_aug) @ (torch.eye(B.shape[1]).to(device) - \
                        X_aug_init[i,:].T @ X_aug_init[i,:] * norm_aug[i,i]**2)) @ V3
    # for j in range(7):
    #     print ((torch.norm(result_list[j+1] - result_list[j]) / torch.norm(result_list[j])).item(), end=', ')
    X = X_init
    X_aug = X_aug_init

    # X = torch.solve(MTTKRP1 - epsilon * (norm @ G @ norm_aug @ X_aug_init).T, prep + torch.eye(B.shape[1]).to(device) * reg)[0].T
    # X_aug = torch.solve(MTTKRP2 - epsilon * (norm_aug @ G @ norm @ X_init).T, prep + torch.eye(B.shape[1]).to(device) * reg)[0].T

    # main step
    tmp_X = torch.einsum('ijkl,ir->rjkl',T_batch,X)
    tmp_X_aug = torch.einsum('ijkl,ir->rjkl',T_batch_aug,X_aug)
    tmp = tmp_X + tmp_X_aug
    prep = X.T @ X + X_aug.T @ X_aug

    A_star = torch.solve(torch.einsum('rjkl,kr,lr->rj',tmp,B,C), \
        prep*(B.T@B)*(C.T@C) + torch.eye(B.shape[1]).to(device) * reg)[0].T
    A = (1-lr) * A + lr * A_star
    B_star = torch.solve(torch.einsum('rjkl,jr,lr->rk',tmp,A,C), \
        prep*(A.T@A)*(C.T@C) + torch.eye(B.shape[1]).to(device) * reg)[0].T
    B = (1-lr) * B + lr * B_star
    C_star = torch.solve(torch.einsum('rjkl,jr,kr->rl',tmp,A,B), \
        prep*(A.T@A)*(B.T@B) + torch.eye(B.shape[1]).to(device) * reg)[0].T
    C = (1-lr) * C + lr * C_star

    rec = torch.norm(torch.einsum('ir,jr,kr,lr->ijkl',X,A,B,C) - T_batch) / torch.norm(T_batch)

    return [A, B, C], rec


