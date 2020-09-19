import numpy as np

class MOESP:    
    def __init__(self,n=2, k = 10,initial_state =  'estimate'):
        self.n = n
        self.k = k
        self.initial_state = initial_state
        
    def LQ(self,A):
        Q_tmp,R_tmp = np.linalg.qr(A.T,mode = 'reduced')
        return np.triu(R_tmp).T, Q_tmp.T

    def  fit(self,u_train,y_train):
        n = self.n
        k = self.k
        step = u_train.shape[1]
        m = (step - k)

        n_u = u_train.shape[2]
        n_y = y_train.shape[2]
        
        N_train = u_train.shape[0]

        U_mat =  np.zeros((n_u*k,m*N_train))
        Y_mat =  np.zeros((n_y*k,m*N_train))
        for i_N in range(N_train):
            U_tmp =  np.zeros((n_u*k,m))
            Y_tmp =  np.zeros((n_y*k,m))
            for i_k in range(k):
                U_tmp[i_k*n_u:(i_k + 1)*n_u,:] = u_train[i_N].T[:,i_k:i_k + m]
                Y_tmp[i_k*n_y:(i_k + 1)*n_y,:] = y_train[i_N].T[:,i_k:i_k + m]
            U_mat[:,i_N *m:(i_N+1)*m] = U_tmp
            Y_mat[:,i_N *m:(i_N+1)*m] = Y_tmp

        W = np.concatenate([U_mat,Y_mat])

        L, Q = self.LQ(W)
        L11 = L[:k*n_u,:k*n_u]
        L21 = L[k*n_u:,:k*n_u]
        L22 = L[k*n_u:,k*n_u:]

        [UU,SS,VV] = np.linalg.svd(L22)

        # calc A,C
        U1 = UU[:,:n]
        Ok = U1.dot(np.diag(np.sqrt(SS[:n])))

        self.A = np.linalg.pinv(Ok[:(k-1)*n_y,:]).dot(Ok[n_y:k*n_y,:])
        self.C = Ok[:n_y,:]


        # calc B,D
        U2 = UU[:,n:]
        M = U2.T.dot(L21).dot(np.linalg.inv(L11))

        XX = np.zeros(((k*n_y-n)*k,n_u))
        RR = np.zeros(((k*n_y-n)*k,n+n_y))
        for i in range(k):
            Li = U2.T[:,n_y*i:n_y*(i+1)]
            bar_Li = U2.T[:,n_y*(i+1):]
            Oki = Ok[:-n_y*(i+1)]
            Ri = np.c_[Li,bar_Li.dot(Oki)]

            Mi = M[:,n_u*i:n_u*(i+1)]
            XX[(k*n_y-n)*i:(k*n_y-n)*(i+1),:] = Mi
            RR[(k*n_y-n)*i:(k*n_y-n)*(i+1),:] = Ri

        DB = np.linalg.pinv(RR).dot(XX)

        self.D = DB[:n_y]
        self.B = DB[n_y:]
 
    # estimate initial state X0
    def predict_initial_state(self,u,y):
        n = self.A.shape[0]
        n_u  =  self.B.shape[1]
        n_y = self.C.shape[0]
        k =  y.shape[0]

        Ok = np.zeros((n_y *k, n))
        for i_k in range(k):
            Ok[i_k*n_y:(i_k+1)*n_y] = self.C @ (np.linalg.matrix_power(self.A,i_k))

        Pk = np.zeros((n_y *k, n_u * k))
        for i_k in range(k):
            Pk[i_k*n_y:(i_k+1)*n_y,i_k*n_u:(i_k+1)*n_u] = self.D
        for i_A in range(k-1):
            for i_k in range(k - i_A-1):
                Pk[(i_A + i_k+1)*n_y:(i_A +i_k+2)*n_y,  (i_k)*n_u:(i_k+1)*n_u] = self.C @np.linalg.matrix_power(self.A,i_A) @self.B

        uk = u[:k,:].T.reshape(-1,1)
        yk = y[:k,:].T.reshape(-1,1)


        x0 = np.linalg.pinv(Ok) @(yk -  Pk.dot(uk))
        return x0
    
    # predict  time series
    def predict(self,x0,u):
        n = self.A.shape[0]
        step =  u.shape[0] 

        x_hat = np.zeros((step,n))
        x_hat[0] = x0.reshape(-1)

        for k in range(step-1):
            x_hat[k+1] = self.A @ x_hat[k] +  self.B @  u[k]

        y_hat = (self.C @ x_hat.T +   self.D @ u.T).T

        return y_hat
    
    #  evaluation of model
    def  score(self,u_test,y_test):
        N_test = u_test.shape[0]
        n = self.A.shape[0]
        error_l2 = np.zeros((N_test))
        for i_N in range(N_test):
            if self.initial_state == 'estimate':
                # 初期値の推定(最初のn点のデータのみ)
                x0 = self.predict_initial_state(u_test[i_N,:2],y_test[i_N,:2])
            if  self.initial_state == 'zero':
                # 初期値は０    
                x0 = np.zeros((n,1)) 

            #  予測 
            y_hat = self.predict(x0,u_test[i_N])
            #  誤差の測定
            error_l2[i_N]  =   np.sqrt(((y_hat - y_test[i_N])**2).sum(1)).sum()/y_hat.shape[0]

        mean_error =  error_l2.sum()/ N_test
        return mean_error
    
    
    
    def autofit(self,u_data,y_data,rate_validation =0.2,n_max = 20):

        # データの分割
        N_train = u_data.shape[0]
        N_validation = round(N_train *rate_validation)
        N_vtrain = N_train  - N_validation
        id_all =  np.random.choice(N_train,N_train,replace=False)
        u_vtrain  =  u_data[id_all[:N_vtrain],:]
        y_vtrain  =  y_data[id_all[:N_vtrain],:]

        u_validation  =  u_data[id_all[N_vtrain:],:]
        y_validation  =  y_data[id_all[N_vtrain:],:]

        step  = u_data.shape[1]
        #  スコアの計算
        k_max = min(n_max * 10, step - n_max)
        score_list = np.inf * np.ones((n_max,k_max))

        #     k>=2n 
        for  n_dim in np.arange(1,n_max):
            for k_dim in np.arange(2 * n_dim, k_max):
                if k_dim+n_dim >=  step:
                    continue;
                self.n = n_dim
                self.k = k_dim
                self.fit(u_vtrain,y_vtrain)
                score_list[n_dim,k_dim] = self.score(u_validation,y_validation)
#                 if 10*score_list[n_dim,k_dim-1] < score_list[n_dim,k_dim] :
#                     break;
                print(self.n,self.k,score_list[n_dim,k_dim])
        self.n = np.where(score_list ==np.min(score_list))[0][0]
        self.k = np.where(score_list ==np.min(score_list))[1][0]
        self.fit(u_train,y_train)
        self.score_list = score_list
        return self.n,self.k

class ORT:
    def __init__(self,n=2, k = 10,initial_state =  'estimate'):
        self.n = n
        self.k = k
        self.initial_state = initial_state
        
    def LQ(self,A):
        Q_tmp,R_tmp = np.linalg.qr(A.T,mode = 'reduced')
        return np.triu(R_tmp).T, Q_tmp.T

    def  fit(self,u_train,y_train):

        n = self.n
        k = self.k
        step = u_train.shape[1]
        m = (step - k)

        n_u = u_train.shape[2]
        n_y = y_train.shape[2]
        
        N_train = u_train.shape[0]

        U_mat =  np.zeros((n_u*k,m*N_train))
        Y_mat =  np.zeros((n_y*k,m*N_train))
        for i_N in range(N_train):
            U_tmp =  np.zeros((n_u*k,m))
            Y_tmp =  np.zeros((n_y*k,m))
            for i_k in range(k):
                U_tmp[i_k*n_u:(i_k + 1)*n_u,:] = u_train[i_N].T[:,i_k:i_k + m]
                Y_tmp[i_k*n_y:(i_k + 1)*n_y,:] = y_train[i_N].T[:,i_k:i_k + m]
            U_mat[:,i_N *m:(i_N+1)*m] = U_tmp
            Y_mat[:,i_N *m:(i_N+1)*m] = Y_tmp


        km = k*n_u //2
        kp = k*n_y //2

        U_tmp = np.zeros((n_u*k,m*N_train))
        U_tmp[:km,:] =  U_mat[km:,:]
        U_tmp[km:,:] =  U_mat[:km,:]
        U_mat = U_tmp


        W = np.concatenate([U_mat,Y_mat])

        L, Q = self.LQ(W)
        L11 = L[:km,:km]
        L41 = L[2*km+ kp:,:km]
        L42 = L[2*km+ kp:,km:2*km]

        [UU,SS,VV] = np.linalg.svd(L42)

        # calc A,C
        U1 = UU[:,:n]
        Ok = U1.dot(np.diag(np.sqrt(SS[:n])))

        self.C = Ok[:n_y,:]
        self.A = np.linalg.pinv(Ok[:-n_y,:]).dot(Ok[n_y:,:])


        # calc B,D
        U2 = UU[:,n:]
        M = U2.T.dot(L41).dot(np.linalg.inv(L11))

        XX = np.zeros(((kp-n)*k//2,n_u))
        RR = np.zeros(((kp-n)*k//2,n+n_y))
        for i in range(k//2):
            Li = U2.T[:,n_y*i:n_y*(i+1)]
            bar_Li = U2.T[:,n_y*(i+1):]
            Oki = Ok[:-n_y*(i+1)]
            Ri = np.c_[Li,bar_Li.dot(Oki)]

            Mi = M[:,n_u*i:n_u*(i+1)]
            XX[(kp-n)*i:(kp-n)*(i+1),:] = Mi
            RR[(kp-n)*i:(kp-n)*(i+1),:] = Ri

        DB = np.linalg.pinv(RR).dot(XX)

        self.D = DB[:n_y]
        self.B = DB[n_y:]

    # estimate initial state X0
    def predict_initial_state(self,u,y):
        n = self.A.shape[0]
        n_u  =  self.B.shape[1]
        n_y = self.C.shape[0]
        k =  y.shape[0]

        Ok = np.zeros((n_y *k, n))
        for i_k in range(k):
            Ok[i_k*n_y:(i_k+1)*n_y] = self.C @ (np.linalg.matrix_power(self.A,i_k))

        Pk = np.zeros((n_y *k, n_u * k))
        for i_k in range(k):
            Pk[i_k*n_y:(i_k+1)*n_y,i_k*n_u:(i_k+1)*n_u] = self.D
        for i_A in range(k-1):
            for i_k in range(k - i_A-1):
                Pk[(i_A + i_k+1)*n_y:(i_A +i_k+2)*n_y,  (i_k)*n_u:(i_k+1)*n_u] = self.C @np.linalg.matrix_power(self.A,i_A) @self.B

        uk = u[:k,:].T.reshape(-1,1)
        yk = y[:k,:].T.reshape(-1,1)


        x0 = np.linalg.pinv(Ok) @(yk -  Pk.dot(uk))
        return x0
    
    # predict  time series
    def predict(self,x0,u):
        n = self.A.shape[0]
        step =  u.shape[0] 

        x_hat = np.zeros((step,n))
        x_hat[0] = x0.reshape(-1)

        for k in range(step-1):
            x_hat[k+1] = self.A @ x_hat[k] +  self.B @  u[k]

        y_hat = (self.C @ x_hat.T +   self.D @ u.T).T

        return y_hat
    
    #  evaluation of model
    def  score(self,u_test,y_test):
        N_test = u_test.shape[0]
        n = self.A.shape[0]
        error_l2 = np.zeros((N_test))
        for i_N in range(N_test):
            if self.initial_state == 'estimate':
                # 初期値の推定(最初のn点のデータのみ)
                x0 = self.predict_initial_state(u_test[i_N,:2],y_test[i_N,:2])
            if  self.initial_state == 'zero':
                # 初期値は０    
                x0 = np.zeros((n,1)) 

            #  予測 
            y_hat = self.predict(x0,u_test[i_N])
            #  誤差の測定
            error_l2[i_N]  =   np.sqrt(((y_hat - y_test[i_N])**2).sum(1)).sum()/y_hat.shape[0]

        mean_error =  error_l2.sum()/ N_test
        return mean_error
    
    
    
    def autofit(self,u_data,y_data,rate_validation =0.2,n_max = 20):

        # データの分割
        N_train = u_data.shape[0]
        N_validation = round(N_train *rate_validation)
        N_vtrain = N_train  - N_validation
        id_all =  np.random.choice(N_train,N_train,replace=False)
        u_vtrain  =  u_data[id_all[:N_vtrain],:]
        y_vtrain  =  y_data[id_all[:N_vtrain],:]

        u_validation  =  u_data[id_all[N_vtrain:],:]
        y_validation  =  y_data[id_all[N_vtrain:],:]

        step  = u_data.shape[1]
        #  スコアの計算
        k_max = min(n_max * 10, step - n_max)
        score_list = np.inf * np.ones((n_max,k_max))

        #     k>=2n 
        for  n_dim in np.arange(1,n_max):
            for k_dim in np.arange(2 * n_dim, k_max,2):
                if k_dim+n_dim >=  step:
                    continue;
                self.n = n_dim
                self.k = k_dim
                self.fit(u_vtrain,y_vtrain)
                score_list[n_dim,k_dim] = self.score(u_validation,y_validation)
#                 if 10*score_list[n_dim,k_dim-1] < score_list[n_dim,k_dim] :
#                     break;
                print(self.n,self.k,score_list[n_dim,k_dim])
        self.n = np.where(score_list ==np.min(score_list))[0][0]
        self.k = np.where(score_list ==np.min(score_list))[1][0]
        self.fit(u_data,y_data)
        self.score_list = score_list
        return self.n,self.k



