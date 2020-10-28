import numpy as np
from scipy.spatial import distance_matrix

import torch
from torch.autograd import Variable
import torch.optim as optim


from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



class ARX:
   def __init__(self,nty=2,ntu=2,N_max = 1000,N_alpha = 10,max_class = 25):
       self.nty = nty # input data
       self.ntu = ntu


   def  fit(self,u_train,y_train):
       self.nt = max(self.ntu,self.nty)
       self.ny = y_train.shape[2]
       self.nu = u_train.shape[2]
       self.nx = self.ntu*self.nu + self.nty*self.ny
       N = y_train.shape[0]*(y_train.shape[1]-self.nt)

       #  入力データの生成
       output_data = np.zeros((N,self.ny))
       for i in range(y_train.shape[0]):
           output_data[i*(y_train.shape[1]-self.nt):(i+1)*(y_train.shape[1]-self.nt),:] =  y_train[i,self.nt:]


       input_data = np.zeros((N,self.nty*self.ny+self.ntu*self.nu))
       for i in range(y_train.shape[0]):
           for j in range(y_train.shape[1]-self.nt):
               input_data[i*(y_train.shape[1]-self.nt)+j] = np.r_[y_train[i,j+self.nt-self.nty:j+self.nt].reshape(-1),u_train[i,j+self.nt-self.ntu:j+self.nt].reshape(-1)]



       # step 学習用のデータの作成
       z_data = np.c_[input_data,np.ones((N,1))]

       self.theta =  np.linalg.pinv(z_data).dot(output_data).T

   def predict_initial_state(self,y):
#          AR type models  do not use u  to predict initial state
       x0  =  y[self.nt-self.nty:self.nt]
       return x0
           
   def predict(self,x0,u):
       y_hat = np.zeros((u.shape[0],x0.shape[1]))
       y_hat[self.nt-self.nty:self.nt] = x0

       for k in range(self.nt,u.shape[0]):
           xk =np.r_[y_hat[k-self.nty:k].reshape(-1),u[k-self.ntu:k].reshape(-1)]
           zk = np.r_[xk,[1]]   
           y_hat[k] =(self.theta*zk).sum(axis = 1)
       return  y_hat
   
   def  score(self,u_test,y_test):
       N_test = u_test.shape[0]
       error_l2 = np.zeros((N_test))
       for i_N in range(N_test):
           x0  =  self.predict_initial_state(y_test[i_N])
           #  予測 
           y_hat = self.predict(x0,u_test[i_N])
           #  誤差の測定
           error_l2[i_N]  =   np.sqrt(((y_hat - y_test[i_N])**2).sum(1)).sum()/y_hat.shape[0]

       mean_error =  error_l2.sum()/ N_test
       return mean_error
   
   def autofit(self,u_train,y_train,rate_validation =0.2,nt_max= 5):

       # データの分割
       N_train = u_train.shape[0]
       N_validation = round(N_train *rate_validation)
       N_vtrain = N_train  - N_validation
       id_all =  np.random.choice(N_train,N_train,replace=False)
       u_vtrain  =  u_train[id_all[:N_vtrain],:]
       y_vtrain  =  y_train[id_all[:N_vtrain],:]

       u_validation  =  u_train[id_all[N_vtrain:],:]
       y_validation  =  y_train[id_all[N_vtrain:],:]

       score_list = np.inf * np.ones((nt_max,nt_max))

       for  i in np.arange(nt_max):
           for  j in np.arange(nt_max):
               self.nty = i+1
               self.ntu = j+1
               self.fit(u_vtrain,y_vtrain)
               score_list[i,j] = self.score(u_validation,y_validation)
               print('nty = {0}, ntu = {1},score= {2}'.format(self.nty,self.ntu,score_list[i,j]))
       self.nty = np.where(score_list ==np.min(score_list))[0][0]+1
       self.ntu = np.where(score_list ==np.min(score_list))[1][0]+1
       self.fit(u_train,y_train)
       self.score_list = score_list
       return self.nty,self.ntu


class PWARX:
    def __init__(self,nty=2,ntu=2,N_max = 1000,N_alpha = 10,max_class = 25):
        self.nty = nty # input data
        self.ntu = ntu
        self.N_max = N_max
        self.N_alpha = N_alpha
        self.max_class  = max_class


    def  fit(self,u_train,y_train,epoch = 10000):
        self.nt = max(self.ntu,self.nty)
        self.ny = y_train.shape[2]
        self.nu = u_train.shape[2]
        self.nx = self.ntu*self.nu + self.nty*self.ny
        N = y_train.shape[0]*(y_train.shape[1]-self.nt)

        #  入力データの生成
        output_data = np.zeros((N,self.ny))
        for i in range(y_train.shape[0]):
            output_data[i*(y_train.shape[1]-self.nt):(i+1)*(y_train.shape[1]-self.nt),:] =  y_train[i,self.nt:]


        input_data = np.zeros((N,self.nty*self.ny+self.ntu*self.nu))
        for i in range(y_train.shape[0]):
            for j in range(y_train.shape[1]-self.nt):
                input_data[i*(y_train.shape[1]-self.nt)+j] = np.r_[y_train[i,j+self.nt-self.nty:j+self.nt].reshape(-1),u_train[i,j+self.nt-self.ntu:j+self.nt].reshape(-1)]


        # データの削減
        N_train = min(N,self.N_max)
        # N_train =N
        index = np.random.randint(0,N,N_train)
        output_train =  output_data[index]
        input_train = input_data[index]

        # step 学習用のデータの作成
        z_data = np.zeros((N,self.ny,self.nx+1))
        for i in range(self.ny):
            z_data[:,i,:] = np.c_[input_data,np.ones((N,1))]

        z_train = z_data[index]

        
        #  kernel の作成
        # kernel に使うデータの作成
        tmp =  np.c_[input_train,output_train]
        # 距離行列の制作
        dist_x = distance_matrix(tmp,tmp)
        # 距離を近い順に N_alpha並べる(対角成分は削除)
        tmp = np.argsort(dist_x,axis=1)[:,1:self.N_alpha+1]
        #   [h_i,t_i ] list化
        indeces_alpha = np.zeros((self.N_alpha*N_train,1))
        for i in range(N_train):
            indeces_alpha[i*self.N_alpha:(i+1)*self.N_alpha,0] = i*np.ones(self.N_alpha)
        indeces_alpha = np.c_[indeces_alpha,tmp.reshape(-1,1)].astype(int)

        
        print("1st step: group lasso by pytorch")
        y_torch = torch.tensor(output_train)
        z_torch = torch.tensor(z_train)

        lambda_omega = 1
        theta_torch = Variable(torch.randn(N_train,self.ny,self.nx+1), requires_grad=True)
        optimizer = optim.Adam([theta_torch],lr = 0.01)


        loss_epoch = np.zeros(epoch)
        MSE_epoch = np.zeros(epoch)
        omega_epoch =np.zeros(epoch)
        for i_epoch in range(epoch):
            #optimizerに使われていた勾配の初期化
            optimizer.zero_grad()
            #omegaの計算  
            omega = torch.norm(theta_torch[indeces_alpha[:,0]].reshape(N_train,-1) - theta_torch[indeces_alpha[:,1]].reshape(N_train,-1),p=1.0,dim=1).mean()

            y_hat = (theta_torch*z_torch).sum(axis = 2)
            MSE=torch.mean(torch.norm(y_torch - y_hat,p=2,dim=1)) 
            #loss関数の計算

            loss = MSE + lambda_omega *omega
            loss_epoch[i_epoch] = loss.item()
            omega_epoch[i_epoch] = omega.item()
            MSE_epoch[i_epoch] = MSE.item()
            #勾配の設定
            loss.backward()
            #最適化の実行
            optimizer.step()
        #出力 
        theta_data = theta_torch.to('cpu').detach().numpy().copy()
        print("2nd step :  Unsupervised Learning")
        aic_n = np.zeros(self.max_class)
        for i in range(self.max_class):
            model =GaussianMixture(n_components=i+1, covariance_type='full',init_params='random')
            model.fit(theta_data.reshape(N_train,-1))
            aic_n[i] = model.aic(theta_data.reshape(N_train,-1))


        n_class = np.argmin(aic_n) + 1
        model =GaussianMixture(n_components=n_class, covariance_type='full',init_params='random')
        model.fit(theta_data.reshape(N_train,-1))
        i_clusters = model.predict(theta_data.reshape(N_train,-1))
        
        print("3rd step : divide input space")
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.clf.fit(input_train, i_clusters)

        print("4th Step : each ARX model training")
        ip_clusters = self.clf.predict(input_data) 
        self.theta = np.zeros((n_class,self.ny,self.nx+1))
        for i_s in range(n_class):
            n_i =sum(ip_clusters==i_s)
            phi_i =  z_data[ip_clusters==i_s,0,:]
            self.theta[i_s] =  np.linalg.pinv(phi_i).dot(output_data[ip_clusters==i_s]).T

    def predict_initial_state(self,y):
#          AR type models  do not use u  to predict initial state
        x0  =  y[self.nt-self.nty:self.nt]
        return x0
            
    def predict(self,x0,u):
        y_hat = np.zeros((u.shape[0],x0.shape[1]))
        y_hat[self.nt-self.nty:self.nt] = x0

        for k in range(self.nt,u.shape[0]):
            xk =np.r_[y_hat[k-self.nty:k].reshape(-1),u[k-self.ntu:k].reshape(-1)]
            i_predict = self.clf.predict(xk.reshape(1,-1))[0]
            zk = np.r_[xk,[1]]   
            y_hat[k] =(self.theta[i_predict] *zk).sum(axis = 1)
        return  y_hat
    
    def  score(self,u_test,y_test):
        N_test = u_test.shape[0]
        error_l2 = np.zeros((N_test))
        for i_N in range(N_test):
            x0  =  self.predict_initial_state(y_test[i_N])
            #  予測 
            y_hat = self.predict(x0,u_test[i_N])
            #  誤差の測定
            error_l2[i_N]  =   np.sqrt(((y_hat - y_test[i_N])**2).sum(1)).sum()/y_hat.shape[0]

        mean_error =  error_l2.sum()/ N_test
        return mean_error
    
    def autofit(self,u_train,y_train,rate_validation =0.2,nt_max= 5):

        # データの分割
        N_train = u_train.shape[0]
        N_validation = round(N_train *rate_validation)
        N_vtrain = N_train  - N_validation
        id_all =  np.random.choice(N_train,N_train,replace=False)
        u_vtrain  =  u_train[id_all[:N_vtrain],:]
        y_vtrain  =  y_train[id_all[:N_vtrain],:]

        u_validation  =  u_train[id_all[N_vtrain:],:]
        y_validation  =  y_train[id_all[N_vtrain:],:]

        score_list = np.inf * np.ones((nt_max,nt_max))

        for  i in np.arange(nt_max):
            for  j in np.arange(nt_max):
                self.nty = i+1
                self.ntu = j+1
                self.fit(u_vtrain,y_vtrain)
                score_list[i,j] = self.score(u_validation,y_validation)
                print('nty = {0}, ntu = {1},score= {2}'.format(self.nty,self.ntu,score_list[i,j]))
        self.nty = np.where(score_list ==np.min(score_list))[0][0]+1
        self.ntu = np.where(score_list ==np.min(score_list))[1][0]+1
        self.fit(u_train,y_train)
        self.score_list = score_list
        return self.nty,self.ntu



